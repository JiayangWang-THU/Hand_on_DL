import torch
# # Create a tensor
# x = torch.arange(12)
# print(x)
# print(x.shape)
# x = x.reshape(3, 4)
# print(x)
# torch.zeros((2, 3, 4))#2大块，每一块3层，每一层4个
# torch.ones((2, 3, 4))
# torch.randn(3, 4)
# torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算
torch.exp(x)  # e的x次幂
# concatenate
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1) #dim表示按哪个维度进行拼接，dim=0就是按最外层的拼接，dim=1就是按第二层的维度进行拼接
X == Y
X.sum()  #所有元素求和
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
a + b #广播机制，自动扩展维度进行计算，a复制行，b复制列
X[-1], X[1:3]#经典左闭右开
X[1, 2] = 9
X #注意此处赋值的索引都是从0开始的所以X[1,2]表示第二行第三列
# 原地操作(in-place operation)用 [:] 来实现    
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
before = id(X)
X += Y #等价于X = X + Y 但是节约了内存
id(X) == before  # True
#可以通过item函数将只有一个元素的张量转换成Python标量
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)