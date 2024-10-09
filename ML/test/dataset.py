# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:24:19 2024

@author: FJJ
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

#%% 训练设备
device = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")
print(f'Using {device} device')
#%% 
"""-------------------------------Fashion-MNIST数据集----------------------------"""
training_data = datasets.FashionMNIST(
    root = 'data',
    train=True,
    download=True,
    transform=ToTensor()
    
    
    )

test_data = datasets.FashionMNIST(
    root='data',
    train = False,
    download=True,
    transform=ToTensor()
    )

#%% 迭代和可视化数据集
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


figure= plt.figure(figsize=(8,8))
cols,rows = 3,3
for i in range(1,cols*rows+1):
    sample_idx = torch.randint(len(training_data),size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap='gray')
plt.show()

figure= plt.figure(figsize=(8,8))
cols,rows = 3,3
for i in range(1,cols*rows+1):
    sample_idx = torch.randint(len(test_data),size=(1,)).item()
    img, label = test_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap='gray')
plt.show()

#%% 使用DataLoaders准备训练数据，以小批量形式传递样本，并重新排列数据减少过拟合，并使用multiprocessing加速数据检索
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data,batch_size =64,shuffle=True)

#%% 迭代dataloader
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
figure = plt.figure()
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.show()
print(f"Label: {label}")

#%% 变换数据使其适合训练
"""所有 TorchVision 数据集都有两个参数 - 
用于修改特征的 transform 和用于修改标签的 target_transform - 
它们接受包含变换逻辑的可调用对象"""
# fashionMNIST特征采用PIL图像格式，标签为整数。需要转换为归一化张量，将标签转换为独热码张量
ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),# 将PIL图像或np.ndarray转化为FloatTensor,并归一化到[0,1]
    target_transform=Lambda(lambda y:torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
    )

#%% 训练神经网络
# 定义神经网络的类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 模型层：获取三张图片
# input_image = torch.rand(3,28,28)

# #初始化nn.Flatten层，将每个图像转换为784个像素值的连续数组
# flatten = nn.Flatten()
# flat_image = flatten(input_image)

# # 线性层：使用其存储的权重和偏差对输入应用线性变换
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())

# #非线性激活函数 nn.ReLU
# hidden1 = nn.ReLU()(hidden1)

# #nn.Sequential 是有序的模块容器，组成神经网络
# seq_modules = nn.Sequential(
#     flatten,
#     layer1,
#     nn.ReLU(),
#     nn.Linear(20,10)
#     )

# 最后一个线性层传递给 nn.Softmax，logits缩放到[0,1]表示模型对每个类的预测概率
# softmax = nn.Softmax(dim=1)
# pred_probab = softmax(logits)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

#%% 反向传播:单层网络
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
loss.backward()
print(w.grad)
print(b.grad)