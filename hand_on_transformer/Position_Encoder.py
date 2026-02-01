"""
Embedding 解决的是：这个 token 是什么
Position Encoding 解决的是：这个 token 在第几个位置
"""
# x = token_embedding(idx) + position_encoding

"""
一个合法的 Position Encoding，必须满足：

用向量来表示位置（能计算绝对位置）
维度 = embedding dim = C
不改变 batch,不改变 token identity
能被加到 [B, T, C] 上

"""

# PE分为科学系的和固定的
# 这里我们只讲固定的
"""
用不同频率的正弦 / 余弦，把“位置”编码成一个连续向量
低维：变化慢（看整体顺序）
高维：变化快（看局部差异）
"""

import torch
import math
def sinusoidal_position_encoding(T, C, device=torch.device("cpu")):
    """
    T: 序列长度
    C: embedding 维度
    return: [T, C]
    """
    pe = torch.zeros(T, C, device=device)
    #arange = arithmetic range，生成一个等差序列
    #unsqueeze(dim) = 在第 dim 个维度插入一个大小为 1 的新维度
    position = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
    # 计算每个维度对应的频率
    div_term = torch.exp(
        torch.arange(0, C, 2, device=device) * (-math.log(10000.0) / C)
    )  # [C/2]
    # 注意这里的步长是2，因为要分别给偶数和奇数维度赋值
    # sinusoidal functions
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
