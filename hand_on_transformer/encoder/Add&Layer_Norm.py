"""
对每一个 token
在 feature 维度 C 上做归一化
[B, T, C] → 对每个 [C] 向量单独归一化
y=LayerNorm(x+f(x))

"""
import torch
import torch.nn as nn
C = 8
ln = nn.LayerNorm(C)
def f(x):
    # 假设 f 是一个简单的两层全连接网络
    return nn.Sequential(
        nn.Linear(C, C*4),
        nn.ReLU(),
        nn.Linear(C*4, C),
    )(x)

x = ln(x + f(x))  # x: [B, T, C]
