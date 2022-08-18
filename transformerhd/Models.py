import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformerhk_v2.Constants as Constants
from transformerhk_v2.Layers import EncoderLayer # , TimeDecoderLayer, TypeDecoderLayer, SetTransformerDecoder
import opt_einsum as oe


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, num_taylor_terms, dropout,
            add_pos=True,
            clamp=None,
            event_update: bool = False,
            explicit_time: bool = False,
            layer_norm: bool = True,
            residual: bool = True,
            decaying: bool = False,  # time decaying
            post_add_pos: bool=True,
            **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.add_pos = add_pos
        self.post_add_pos = post_add_pos

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        event_emb_mask = torch.ones(num_types + 1, 1)
        event_emb_mask[Constants.PAD] = 0
        self.register_buffer("event_emb_mask", event_emb_mask)
        # self.event_emb = nn.Embedding(num_types, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         num_taylor_terms, dropout=dropout,
                         normalize_before=True, clamp=clamp,
                         event_update=event_update,
                         explicit_time=explicit_time,
                         layer_norm=layer_norm,
                         residual=residual,  # layer_norm outside
                         decaying=decaying,
                         **kwargs)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask, epoch=None):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        enc_output = self.event_emb(event_type)
        event_emb = self.event_emb.weight  # might be OOM when number of event-type is large
        if self.event_emb_mask is not None:
            event_emb = event_emb * self.event_emb_mask
        event_type = F.one_hot(event_type, num_classes=event_emb.size(0)).type_as(event_emb)
        for enc_layer in self.layer_stack:
            if self.add_pos:
                enc_output += tem_enc
            enc_output, _, event_emb = enc_layer(
                enc_output,
                event_emb,
                event_type,
                tem_enc,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                epoch=epoch
            )
            if self.event_emb_mask is not None:
                event_emb = event_emb * self.event_emb_mask

        if self.post_add_pos:
            enc_output += tem_enc

        return enc_output, event_emb



class PredictorTime(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim):
        super().__init__()

        # self.linear = nn.Linear(dim, 1, bias=False) # fixme; original
        self.linear = nn.Linear(dim, 1, bias=True)
        nn.init.xavier_normal_(self.linear.weight)
        self.sp = nn.Softplus()

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = self.sp(out)
        out = out * non_pad_mask

        return out


class PredictorTypes(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class InnerTypes(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, event_emb, non_pad_mask):
        hid = self.linear(data)
        out = oe.contract("bsd, yd->bsy", hid, event_emb, backend="torch")
        select_mask = torch.ones(event_emb.size(0), device=event_emb.device, dtype=torch.bool)
        select_mask[Constants.PAD] = False
        out = out[..., select_mask]
        out = out * non_pad_mask
        return out


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask

        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        # self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        # data = torch.flip(data, [0, 1])
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]
        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, num_taylor_terms=5,
            dropout=0.1, clamp=None,
            event_update: bool = False,
            explicit_time: bool = False,
            transf_dec: bool = False,
            inner_type: bool = False,
            decaying: bool = True,  # effect decaying as time interval large.
            # inner_type: bool = True,
            **kwargs):
        '''
        :param num_types:
        :param d_model:
        :param d_rnn:
        :param d_inner:
        :param n_layers:
        :param n_head:
        :param d_k:
        :param d_v:
        :param num_taylor_terms:
        :param dropout:
        :param clamp:
        :param event_update:  update event emb inside each transformer layer
        :param explicit_time:  using explicit time encoder (non-stationary)
        :param kwargs:
        '''
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            num_taylor_terms=num_taylor_terms,
            dropout=dropout,
            clamp=clamp,
            event_update=event_update,
            explicit_time=explicit_time,
            decaying=decaying,
            **kwargs
        )

        self.num_types = num_types
        self.inner_type = inner_type
        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)


        # Fixme: (Aug 2nd)  ---> figure out what `self.alpha` and `self.beta` for
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        # Decoder
        # OPTIONAL recurrent layer, this sometimes helps
        self.transf_dec = transf_dec
        self.transf_dec = False
        self.no_rnn = True
        if self.transf_dec:
            pass
            # self.dec = SetTransformerDecoder(d_model, d_inner, n_head,  # d_k, d_v,
            #                                  layer_norm=True, residual=True)
        else:
            self.dec = RNN_layers(d_model, d_rnn)
        # prediction of next time stamp
        self.time_predictor = PredictorTime(d_model)

        self.type_predictor = PredictorTypes(d_model, num_types)
        if self.inner_type:
            self.type_predictor = InnerTypes(d_model, d_model)

    def forward(self, event_type, event_time, epoch=None):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output, event_emb = self.encoder(event_type, event_time, non_pad_mask, epoch=epoch)

        # Decoder
        # Can improve
        if self.transf_dec:
            dec_output = self.dec(enc_output, non_pad_mask=non_pad_mask)
        else:
            dec_output = self.dec(enc_output, non_pad_mask)
        if self.no_rnn:
            dec_output = enc_output

        time_prediction = self.time_predictor(dec_output, non_pad_mask)

        # if self.inner_type:
        #     type_prediction = self.type_predictor(dec_output, event_emb,  non_pad_mask)
        # else:
        type_prediction = self.type_predictor(dec_output, non_pad_mask)
        return dec_output, (type_prediction, time_prediction)


