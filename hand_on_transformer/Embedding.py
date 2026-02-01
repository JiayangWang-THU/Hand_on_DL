import torch
import torch.nn as nn
# 这一章主要讲embedding，虽然想用jupyter notebook写
# 但实在太多代码块了，不方便管理，就直接用py文件写了
"""
Tokenization 决定“编号怎么来”，Embedding 决定“编号怎么用”。

"""
B = 2      # batch size
T = 4      # sequence length
C = 8      # embedding dimension
V = 10     # vocab size
"""
B 个句子
每个句子 T 个 token
每个 token 是一个 C 维向量
"""


"""
Tokenization = 把“原始文本”切分并映射成 token id
"我爱你"  →  ["我", "爱", "你"]  →  [17, 42, 9]
这一步的产物只有一个东西：
idx  # 整数张量
"""
idx = torch.tensor([
    [1, 2, 3, 4],   # 第一句
    [4, 3, 2, 1],   # 第二句
])  # shape: [B, T]


# Embedding = 把“token id”映射成“token 向量”
embedding = nn.Embedding(V, C)

x = embedding(idx)
print(x.shape)
print(embedding.weight)
"""
Embedding 学到的是一种几何布局：
相似 token → 向量距离近
不同 token → 向量距离远
"""