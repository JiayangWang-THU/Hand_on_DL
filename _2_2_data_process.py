import os

os.makedirs(os.path.join('data'), exist_ok=True) # 创建存放数据的文件夹
data_file = os.path.join('data', 'house_tiny.csv') # 数据文件路径
with open(data_file, 'w') as f: # 写入数据
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
import pandas as pd
data = pd.read_csv(data_file)
print(data)
#处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]#输入是所有行前两列，输出是第三列 iloc是取整数位置
inputs = inputs.fillna(inputs.mean(numeric_only=True))#只对列的求和取平均填充NaN
inputs = pd.get_dummies(inputs, dummy_na=True)#将类别变量转换为指示变量
print(inputs)
import torch
#转化为numpy数组再转化为张量
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)