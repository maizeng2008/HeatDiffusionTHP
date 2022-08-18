import torch
import torch.nn as nn
from torch.nn import functional as F

from transformerhk_v2.SubLayers import HeatDiffusionAttention, PositionwiseFeedForward
from transformerhk_v2.Modules import ScaledDotProductAttention
import opt_einsum as oe


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, num_taylor_terms,
                 dropout=0.1,
                 normalize_before=True,
                 clamp=None,
                 event_update: bool = False,
                 explicit_time: bool = False,
                 layer_norm: bool = True,
                 residual: bool=True,
                 decaying:bool=False,
                 # pre_layer_norm: bool = True,
                 **kwargs):
        super(EncoderLayer, self).__init__()

        # layer_norm outside instead of inside the module.
        self.slf_attn = HeatDiffusionAttention(
            n_head, d_model, d_k, d_v, num_taylor_terms,
            dropout=dropout, normalize_before=normalize_before,
            clamp=clamp,
            event_update=event_update,
            explicit_time=explicit_time,
            layer_norm=False,  # done outside attention
            residual=False,
            decaying=decaying,
            **kwargs
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout,
            normalize_before=normalize_before,
            layer_norm=False, # done outside
        )

        self.O_h = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.O_h.weight)

        self.layer_norm = layer_norm
        self.residual = residual
        if layer_norm:
            self.ln_1 = nn.LayerNorm(d_model)
            self.ln_2 = nn.LayerNorm(d_model)

        self.event_update = event_update
        if event_update:
            self.pos_ffn_event = PositionwiseFeedForward(
                d_model, d_inner, dropout=dropout,
                normalize_before=normalize_before,
                layer_norm=False,  # done outside
            )
            if self.layer_norm:
                self.ln_event = nn.LayerNorm(d_model)

        self.normalize_before = normalize_before


    def forward(self, enc_input, event_emb, event_type, enc_time, non_pad_mask=None, slf_attn_mask=None, epoch=None):

        if self.layer_norm and self.normalize_before:
            enc_input = self.ln_1(enc_input.flatten(2))

        if self.layer_norm and self.normalize_before and self.event_update:
            event_emb = self.ln_event(event_emb)

        enc_output, enc_slf_attn, _ = self.slf_attn(
            event_emb, event_emb, enc_input, enc_time, event_type, mask=slf_attn_mask, epoch=epoch)

        enc_output = self.O_h(enc_output.flatten(2))
        if self.residual:
            enc_output += enc_input.flatten(2)
        if self.layer_norm and (not self.normalize_before):
            enc_output = self.ln_1(enc_output)

        enc_output *= non_pad_mask

        enc_input = enc_output
        if self.layer_norm and self.normalize_before:
           enc_output = self.ln_2(enc_output)

        enc_output = self.pos_ffn(enc_output)
        if self.residual:
            enc_output += enc_input.flatten(2)
        if self.layer_norm and (not self.normalize_before):
            enc_output = self.ln_2(enc_output)

        enc_output *= non_pad_mask

        if self.event_update:
            event_out = self.pos_ffn_event(event_emb)
            if self.residual:
                event_out += event_emb
            if self.layer_norm and (not self.normalize_before):
                event_out = self.ln_event(event_out)
            event_emb = event_out

        return enc_output, enc_slf_attn, event_emb











