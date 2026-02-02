"""
Decoder 的 masked self-attention 本质是在做：
「在不知道未来的前提下，用已经生成的历史，预测下一个 token。」
预测第 t 个词
只能用 t-1 之前的词
"""
import torch
# causal mask的手搓
# 注意为什么mask是T*T的，因为我们是算的token之间的关系，qk相乘也得到的是T*T的矩阵
# 最后的结果应该是上三角矩阵被mask掉
def causal_mask(T, device=None):
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    for i in range(T):
        for j in range(T):
            if j > i:
                mask[i, j] = True
    return mask # [T, T]
x = torch.randn(2, 4, 8)  # [B, T, C]
W_Q = torch.nn.Linear(8, 8)
W_K = torch.nn.Linear(8, 8)
W_V = torch.nn.Linear(8, 8)
Q = W_Q(x)   # [B, T, C]
K = W_K(x)   # [B, T, C]
V = W_V(x)   # [B, T, C]
def masked_self_attention(Q, K, V,H):
    T = Q.shape[1]
    B = Q.shape[0]
    C = Q.shape[2]
    
    assert C % H == 0
    d_k = C // H  
    
    # [B,T,C] -> [B,H,T,d_k]
    Q = Q.view(B, T, H, d_k).transpose(1, 2)
    K = K.view(B, T, H, d_k).transpose(1, 2)
    V = V.view(B, T, H, d_k).transpose(1, 2) # [B, T, T]

    from math import sqrt
    mask = causal_mask(T, device=Q.device)
    # 用 mask 去屏蔽未来位置
    scores = (Q @ K.transpose(-2, -1)) / (sqrt(d_k))  # [B, H, T, T]
    mask = causal_mask(T, device=Q.device).unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
    scores = scores.masked_fill(mask, float('-inf'))

    scores = torch.softmax(scores, dim=-1)
    out = scores @ V
    # 拼回 [B,T,C]
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return out
