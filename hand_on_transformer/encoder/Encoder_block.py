from torch import nn

class Encoder_Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.add_norm1 = AddNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x):
        mha_out = self.mha(x)
        mha_out = self.dropout1(mha_out)
        x = self.add_norm1(x, mha_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.add_norm2(x, ffn_out)
        return x

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.ln(x + sublayer_output)

class MHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        out, _ = self.mha(x, x, x, need_weights=False)
        return out

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)
