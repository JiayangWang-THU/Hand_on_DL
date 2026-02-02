import torch
import math
from torch import nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MaskedMHA(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = CrossMHA(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FFN(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_key_padding_mask=None, src_key_padding_mask=None):
        # 1) masked self-attn
        out1 = self.self_attn(x, tgt_key_padding_mask=tgt_key_padding_mask)
        out1 = self.dropout1(out1)
        x = self.add_norm1(x, out1)

        # 2) cross-attn
        out2 = self.cross_attn(x, enc_out, src_key_padding_mask=src_key_padding_mask)
        out2 = self.dropout2(out2)
        x = self.add_norm2(x, out2)

        # 3) ffn
        out3 = self.ffn(x)
        out3 = self.dropout3(out3)
        x = self.add_norm3(x, out3)

        return x


class CrossMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.C = d_model
        self.H = num_heads
        self.d = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x_dec, enc_out, src_key_padding_mask=None):
        """
        x_dec:   [B, T_tgt, C]
        enc_out: [B, T_src, C]
        src_key_padding_mask: [B, T_src] True 表示 PAD
        """
        B, T_tgt, C = x_dec.shape
        T_src = enc_out.shape[1]

        Q = self.W_Q(x_dec)
        K = self.W_K(enc_out)
        V = self.W_V(enc_out)

        Q = Q.view(B, T_tgt, self.H, self.d).transpose(1, 2)  # [B,H,Tt,d]
        K = K.view(B, T_src, self.H, self.d).transpose(1, 2)  # [B,H,Ts,d]
        V = V.view(B, T_src, self.H, self.d).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d)  # [B,H,Tt,Ts]

        if src_key_padding_mask is not None:
            pmask = src_key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,Ts]
            scores = scores.masked_fill(pmask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # over source positions
        attn = self.attn_dropout(attn)

        out = attn @ V  # [B,H,Tt,d]
        out = out.transpose(1, 2).contiguous().view(B, T_tgt, C)
        out = self.W_O(out)
        out = self.out_dropout(out)
        return out

class MaskedMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.C = d_model
        self.H = num_heads
        self.d = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_key_padding_mask=None):
        """
        x: [B, T, C]
        tgt_key_padding_mask: [B, T] True 表示 PAD
        """
        B, T, C = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.H, self.d).transpose(1, 2)  # [B,H,T,d]
        K = K.view(B, T, self.H, self.d).transpose(1, 2)
        V = V.view(B, T, self.H, self.d).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d)  # [B,H,T,T]

        # causal mask: [1,1,T,T]
        cmask = causal_mask(T, device=x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(cmask, float("-inf"))

        # padding mask: [B,1,1,T]
        if tgt_key_padding_mask is not None:
            pmask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(pmask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ V  # [B,H,T,d]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B,T,C]
        out = self.W_O(out)
        out = self.out_dropout(out)
        return out

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        return self.ln(x + sublayer_out)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)



def causal_mask(T, device=None):
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    for i in range(T):
        for j in range(i + 1, T):
            mask[i, j] = True   # 未来位置屏蔽
    return mask
