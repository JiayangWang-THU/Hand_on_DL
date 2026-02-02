"""
cross-attention
一般出现在(Encoder-Decoder / Seq2Seq)

常见场景为
机器翻译 source=英文句子 target=中文句子
摘要 source=长文章 target=摘要
任务型生成 source=指令/表格/代码片段 target=输出
传统 seq2seq(T5、BART 这类)


❌ 不需要 cross-attention(Decoder-only / GPT 类)
如果模型的“条件”已经直接放进了同一条序列里
prompt + answer 拼一起，那就不需要单独的 encoder / cross-attn

GPT [prompt tokens][answer tokens] 全在一个序列里

这时模型只靠 masked self-attention 就能“既看 prompt 又看已生成历史”
一句话：
如果条件信息已经在 decoder 的 token 序列里了，就不需要 encoder / cross-attn

关键
实际上encoder提供的是“条件信息”
encoder 的 self-attention 不需要 causal mask 
因为 source 是给定的、完整可见的
读文章当然可以看全文!!
"""

"""
Q 来自 decoder 上一层
x = [B, T_dec, C]

K,V 来自 encoder
enc_out = [B, T_enc, C]

Scores = T_tgt * T_src (每个 target token 看所有 source token 的相关性)
"""

import torch

x = torch.randn(2, 4, 8)  # decoder 输入 [B, T_dec, C]
enc_out = torch.randn(2, 6, 8)  # encoder 输出 [B, T_enc, C]
W_Q = torch.nn.Linear(8, 8)
W_K = torch.nn.Linear(8, 8) 
W_V = torch.nn.Linear(8, 8)
Q = W_Q(x)           # [B, T_dec, C]
K = W_K(enc_out)    # [B, T_enc, C]
V = W_V(enc_out)    # [B, T_enc, C]
import math
def cross_attention(Q, K, V , H,src_key_padding_mask=None):
    T_dec = Q.shape[1]        # T_dec
    T_end = K.shape[1]        # T_enc
    assert Q.shape[2] % H == 0
    d_k = Q.shape[2] // H  # C // H
    # 切分多头
    Q = Q.view(Q.shape[0], T_dec, H, d_k).transpose(1, 2)   # [B, H, T_dec, d_k]
    K = K.view(K.shape[0], T_end, H, d_k).transpose(1, 2)   # [B, H, T_enc, d_k]
    V = V.view(V.shape[0], T_end, H, d_k).transpose(1, 2)   # [B, H, T_enc, d_k]

    # 计算注意力得分
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)  # [B, T_dec, T_enc]

    if src_key_padding_mask is not None:
        # [B, T_src] -> [B, 1, 1, T_src]，广播到所有 head 和所有 tgt 位置
        mask = src_key_padding_mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)              # [B, T_tgt, T_src]
    out  = attn @ V                                   # [B, T_tgt, d]
    out = out.transpose(1, 2).contiguous().view(out.shape[0], T_dec, -1)  # [B, T_dec, C]
    return out
out = cross_attention(Q, K, V, H=2)
print(out.shape)