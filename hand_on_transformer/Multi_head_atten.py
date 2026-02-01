"""
Q = 我现在想找什么
K = 我这里有什么
V = 真正被拿走的信息
对序列中每一个位置,让它“看一眼”其它位置(Q @ K)，并聚合有用的信息(P @ V)


注意力 = 对一堆信息，按“相关性”做加权平均
向量相似度 = 点积
因为线性相关，最大的时候叫共线，最小的时候叫正交
更相似的向量，权重更大
Q/K/V 不是为了复杂，而是为了“解耦三个角色”
"""


"""
Q、K、V 不是新数据，
它们是从同一个输入 X 通过不同线性映射得到的。
"""
import torch
import torch.nn as nn
B, T, C = 2, 4, 8

X = torch.randn(B, T, C)  # 假设这是 embedding + PE 的输出
# nn.Linear(C, C) 永远只作用在最后一维上
W_Q = nn.Linear(C, C, bias=False)
W_K = nn.Linear(C, C, bias=False)
W_V = nn.Linear(C, C, bias=False)
"""
实际的内部逻辑是这样的：
for b in range(B):
    for t in range(T):
        Q[b, t] = X[b, t] @ W_Q.weight.T

每一个 X[b, t] 是一个 [C] 向量
W_Q.weight.T 是 [C, C]
这是标准的 向量 @ 矩阵 
X[b, t]      : [1 × C]
W_Q.weight.T : [C × C]
"""
Q = W_Q(X)   # [B, T, C]
K = W_K(X)   # [B, T, C]
V = W_V(X)   # [B, T, C]

#print(Q.shape, K.shape, V.shape)


# 单头 Attention
import math
def single_head_attention(Q, K, V):
    """
    Q, K, V: [B, T, C]
    return: [B, T, C]
    """
    d = Q.shape[-1]

    # 1. 相似度打分
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    # scores: [B, T, T]

    # 2. softmax
    attn = torch.softmax(scores, dim=-1)

    # 3. 加权求和
    out = attn @ V  # [B, T, C]
    return out
# out = single_head_attention(Q, K, V)
# print(out.shape)
# print(out)

# 多头 Attention
"""
一个注意力头太“单一”，
多个头可以从不同子空间看序列（不同的投影，不同的相关性计算方法）。

工程上做法：
把 C 拆成 H 份
每个 head 维度 d=C/H

多头 ≠ 多个注意力算子
多头 = 同一个注意力算子，在不同子空间并行看序列
每个 head 有自己的一套 Q/K/V 投影矩阵

例如：
head 1：按“语义相似”看关系
head 2：按“句法依赖”看关系
head 3：按“远近上下文”看关系

为什么用“切 C 维”而不是复制？
因为：
计算量不能爆炸
参数量要可控
最后还能 concat 回原维度
"""

# C -> H and d
H = 2
d = C // H
Qh = Q.view(B, T, H, d).transpose(1, 2)
Kh = K.view(B, T, H, d).transpose(1, 2)
Vh = V.view(B, T, H, d).transpose(1, 2)

def multi_head_attention(Q, K, V):
    """
    Q, K, V: [B, H, T, d]
    return: [B, H, T, d]
    """
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out

# [B, H, T, d] → [B, T, C]
# 注意多头计算完了之后不是立马相加，不然会丢失刚才计算出来的特征数
# 保留每个 head 的独立信息，而不是立刻混掉
# 所以先 concat 回去，再用一个线性层把维度变回 C
out_h = multi_head_attention(Qh, Kh, Vh)
concat = out_h.transpose(1, 2).contiguous().view(B, T, C)
W_O = nn.Linear(C, C, bias=False)
out = concat @ W_O
"""
X ∈ [B, T, C]
 ↓ Linear
Q, K, V
 ↓ reshape
[B, H, T, d]
 ↓ attention
[B, H, T, d]
 ↓ concat
[B, T, C]
"""

