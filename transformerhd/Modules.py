import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import opt_einsum as oe
# from MLP import MLP
# import math

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale_constant=None, attn_dropout=0., clamp=None):
        super().__init__()

        # self.scale_constant = scale_constant
        self.dropout = nn.Dropout(attn_dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None

    def forward(self, q, k, mask=None):
        # h x lq x dv      h x lq x dv
        # q = q / self.scale_constant
        q = q / np.sqrt(q.size(-1))
        if q.dim() == 3:
            attn = oe.contract("hqd, hkd-> hqk", q, k, backend="torch")
        elif q.dim() == 4:
            attn = oe.contract("bhqd, bhkd-> bhqk", q, k, backend="torch")
        else:
            raise NotImplementedError(f"dim [{q.dim()}] is not supported yet")

        if self.clamp is not None:
            attn = torch.clamp(attn, -self.clamp, self.clamp)

        if mask is not None:
            attn = attn.masked_fill_(mask, -1e16)

        attn = self.dropout(F.softmax(attn, dim=-1))
        return attn
        # h x lq x lq
