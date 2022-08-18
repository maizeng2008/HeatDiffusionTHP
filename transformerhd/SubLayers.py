import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformerhkhd.Constants as Constants
from transformerhkhd.Modules import ScaledDotProductAttention
from transformerhkhd.MLP import MLP
import opt_einsum as oe
import transformerhkhd.Constants as Constants
from pathlib import Path

class HeatDiffusionAttention(nn.Module):
    """
    $X^{(t)} = e^{(A-I)t} X^{(0)}$


    > Attention based on event types
    - q, k: event type representation
    - v:  event representation

    - to expand from event types to events
    - update events
    """

    def __init__(self, n_head, d_model, d_k, d_v, num_taylor_terms, dropout=0.1,
                 normalize_before=False,
                 weight_tying=False,
                 clamp=None,
                 event_update: bool = False,
                 explicit_time: bool = False,
                 layer_norm=False,
                 residual=False,
                 output_proj=False,
                 decaying=True,
                 explicit_decay=False,
                 **kwargs):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.explicit_time = explicit_time
        self.event_update = event_update
        self.layer_norm = layer_norm
        self.residual = residual
        self.decaying = decaying
        self.visual = False
        if self.decaying:
            self.gh_decay = nn.Linear(d_model, n_head, bias=True)
            nn.init.xavier_normal_(self.gh_decay.weight)

        if self.decaying:
            self.gh_decay = nn.Linear(d_model, n_head, bias=True)
            nn.init.xavier_normal_(self.gh_decay.weight)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        #### Heat Kernel Code ####
        if self.explicit_time:
            self.gh_1 = nn.Linear(d_model, n_head, bias=True)
            self.gh_2 = nn.Linear(d_model, n_head, bias=False)
            self.gh_3_w = nn.Parameter(torch.zeros(d_k, 1, n_head), requires_grad=True)
            self.gh_3_b = nn.Parameter(torch.zeros(1, 1, 1, n_head), requires_grad=True)
            nn.init.xavier_uniform_(self.gh_1.weight)
            nn.init.xavier_uniform_(self.gh_2.weight)
            nn.init.xavier_uniform_(self.gh_3_w)
        else:
            self.gh_1 = nn.Linear(d_model, n_head, bias=True)
            nn.init.xavier_uniform_(self.gh_1.weight)

        self.num_taylor_terms = num_taylor_terms
        ft = [1.]
        for i in range(num_taylor_terms - 1):
            ft.append(ft[-1] * (i + 1))

        ft = torch.Tensor(ft)
        for _ in range(4):
            ft = ft.unsqueeze(0)

        self.register_buffer("taylor", torch.Tensor(ft))  # (k,)

        #### Heat Kernel Code ####
        # if output_proj:
        #     self.fc = nn.Linear(d_v * n_head, d_model)
        #     nn.init.xavier_uniform_(self.fc.weight)
        # else:
        #     self.fc = nn.Identity()
        #
        # self.event_update = event_update
        # if not event_update:
        #     self.fc_emb = nn.Identity()
        # else:
        #     self.fc_emb = nn.Linear(d_model, d_model)
        #     self.act = nn.GELU()
        #     self.ln_event = nn.LayerNorm(d_model, eps=1e-6)

        self.attention = ScaledDotProductAttention(scale_constant=d_k ** 0.5, attn_dropout=dropout, clamp=clamp)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        if not layer_norm:
            self.layer_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        self.softplus_time = True
        self.debug = False

    def forward(self, q, k, v, z, y, mask=None, epoch=None):
        '''
        :param q:  query  (event type repr) (b, lq, dv)
        :param k:  key    (event type repr) (b, lk=lq, dv)
        :param v:  value  (event repr)      (b, lv, dv)
        :param z:  time-encoding            (b, lv, dv)
        :param y:  event indicator          (b, lk, lq)  # one-hot
        :param mask:  event masking         (b, lv x lv)
        :return:
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        residual_q = q
        # remove the redundant embedding for attn

        pad_mask = torch.ones(q.size(0), device=q.device, dtype=torch.bool)
        pad_mask[0] = False
        q = q[pad_mask]
        k = k[pad_mask]

        sz_b, len_q, len_k, len_v = v.size(0), q.size(0), k.size(0), v.size(1)

        residual = v
        if self.normalize_before:
            v = self.layer_norm(v)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(len_q, n_head, d_k)
        k = self.w_ks(k).view(len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv ,  b x n x lk x dv
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(1, 2)
        # q,k:  (h x lq x d_k)
        # v :   (b x h x lv x d_v)
        # v = v.transpose(1, 2)

        attn = self.attention(q, k, mask=None)  # do masking outside
        
        # h x lq x lq
        I = torch.eye(len_q, device=attn.device).unsqueeze(0)
        attn = attn - I
        attn = F.pad(attn, (1, 0, 1, 0), value=0)


        # decay_matrix = torch.zeros_like(attn)
        # decay_matrix[..., 0] = 1.

        if epoch is not None and self.debug:
            if epoch == 20:
                breakpoint()

        # b x n x lq x lk=lq
        # (b, lq, dv) ---> (b, lq, 1, dv) X (b, 1, lq, dv) ---> (b, lq, lq, dv)
        # Gh = torch.exp(self.gh_1(z.unsqueeze(2)) + self.gh_2(z.unsqueeze(1))) # time-enc --> logit-time --> time
        # Gh = F.softplus(self.gh_1(z.unsqueeze(2)) + self.gh_2(z.unsqueeze(1))) # time-enc --> logit-time --> time
        if not self.explicit_time:
            # Gh = F.softplus(self.gh_1(z.unsqueeze(2) - z.unsqueeze(1))) # time-enc --> logit-time --> time
            Gh = self.gh_1(z.unsqueeze(2) - z.unsqueeze(1))  # time-enc --> logit-time --> time
        else:
            left = self.gh_1(z)
            right = self.gh_2(z)
            gh = left.unsqueeze(2) - right.unsqueeze(1)  # (b, lq, lk, dv)
            gh = gh.view(gh.size(0), gh.size(1), gh.size(2), self.gh_3_w.size(-1), -1)
            Gh = oe.contract("bqkhd, dch -> bqkhc", gh, self.gh_3_w, backend="torch").squeeze(-1) + self.gh_3_b

        if self.softplus_time:
            Gh = F.softplus(Gh)

        # (b, lq, lq, dv) --> (b, lq, lq, n)
        Gh = Gh.permute(0, 3, 1, 2)  # (-1, n, lq, lq)

        hks = []
        ghs = []
        attn_ = attn
        Gh_ = Gh
        for i in range(self.num_taylor_terms):
            hks.append(attn_)
            ghs.append(Gh_)
            if self.softplus_time:
                Gh_ = Gh_ * Gh
            else:
                Gh_ = Gh_ + Gh

            attn_ = torch.bmm(attn_, attn)

        ## ------------- einsum implementation ---------------
        hks = torch.stack(hks, dim=1)  # hks: (h, tl, lq, lk)
        hks = oe.contract("htqk, bvq-> bhtvk", hks, y, backend="torch")
        hks = oe.contract("bhtvk, bok-> bhtvo", hks, y, backend="torch")

        if self.softplus_time:
            ghs = torch.stack(ghs, dim=2) / self.taylor.permute(0, 1, 4, 2, 3)
        else:
            # todo: for exp based time output with `log-sum-exp` tricks
            ghs = torch.exp(torch.stack(ghs, dim=2)) / self.taylor.permute(0, 1, 4, 2,
                                                                           3)  # if use exp  (log-sum-exp trick)
        ## ------------- einsum implementation ---------------

        # (b, h, ty, lq, lq)
        T = oe.contract("bhtvo, bhtvo->bhvo", hks, ghs, backend="torch")
        if self.decaying:
            decay = F.softplus(self.gh_decay(z.unsqueeze(2) - z.unsqueeze(1))).permute(0, 3, 1, 2)
            T = T * torch.exp(-decay)
            # exponential time decaying factor (exp(-\sigma(t-t_h))
            # expect that larger time-interval has stronger decay effect

        # T = (hks * ghs).sum(dim=2)
        # ------------- einsum implementation ---------------


        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
            T = T.masked_fill_(mask, 0)  # no exp --> mask with 0 instead of -1e16

            # fixed: just fixed from -1e16 to 0
            # fixme: (Aug 1) see which shape of the mask (hadamard product or masked_fill)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # v: b x n x lv x dv    T: b x n x lv x lv

        output = oe.contract("bhvo, bhod-> bhvd", T, v) + v
        # fixed: add v as the 0th order term in Taylor Approximation;

        # output: (b x n x lv x dv)  --> (b x lv x n x dv)
        output = output.transpose(1, 2)
        output = output.flatten(2)
        output = self.dropout(output)
        # output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # output = self.dropout(self.fc(output))
        if self.residual:
            output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)

        event_emb = residual_q

        return output, attn_, event_emb


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True, layer_norm=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        if not layer_norm:
            self.layer_norm = nn.Identity()
        else:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True, layer_norm=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        if not layer_norm:
            self.layer_norm = nn.Identity()
        else:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


