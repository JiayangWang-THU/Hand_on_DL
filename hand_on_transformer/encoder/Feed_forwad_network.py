"""
FFN = 对每一个 token，独立地做一次“非线性特征变换”。
逐 token
不看别的 token
引入非线性

作用：
Attention 本身几乎是线性的（softmax + 加权和）
FFN 引入非线性，使模型能够学习更复杂的特征组合
FFN(x)=max(0,x @ W1+ b1) @ W2 + b2 
max(0,·) 是 ReLU 激活函数
[B, T, C]
 ↓ FFN
[B, T, C]
对每一个 token 向量 [C]
单独过一个两层 MLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
B, T, C = 2, 4, 8
"""
把维度放大，相当于：
给模型一个“高维工作台”
在高维空间里做非线性组合
再压回原维度
"""
d_ff = 32   # 通常是 4 * C

x = torch.randn(B, T, C)

W1 = nn.Linear(C, d_ff)
W2 = nn.Linear(d_ff, C)

# FFN 前向
h = W1(x)           # [B, T, d_ff]
h = F.gelu(h)       # 非线性
out = W2(h)         # [B, T, C]

print(out.shape)
print(out)

