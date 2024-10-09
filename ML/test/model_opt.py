# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:18:34 2024

@author: FJJ
"""

import torch
import torchvision.models as models
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt



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

#定义训练循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)
        # 训练循环三部分
        loss.backward() #获取损失函数对每个参数的梯度
        optimizer.step() #根据反向传播中收集的梯度调整参数
        optimizer.zero_grad() # 重置模型梯度为0
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size+len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss , correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            
    return test_loss, (100*correct)

#%%
if __name__=='__main__':
    
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    model = NeuralNetwork()
    
    # 超参数： 可调整的参数，影响模型训练和收敛速度
    learning_rate = 1e-3
    batch_size = 64
    
    #损失函数：nn.MSELoss(均方误差，用于回归)，nn.NLLLoss(负对数似然，用于分类),nn.CrossEntropyLoss结合二者
    loss_fn = nn.CrossEntropyLoss()
    
    #优化器
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    epochs = 5
    
    #%% 模型训练
    loss_list = []
    correct_list = []
    for t in range(epochs):
        print(f"Epoch{t+1}\n")
        train_loop(train_dataloader,model,loss_fn,optimizer)
        
        loss_value, correct_value =test_loop(test_dataloader, model, loss_fn)
        loss_list.append(loss_value)
        correct_list.append(correct_value)
        
    print('Done!\n')
    
    # #%% 保存和加载模型权重
    # torch.save(model.state_dict(),'model_weights.pth')
    # # 要加载权重，需要先创建相同实例
    # model_load = NeuralNetwork()
    # model_load.load_state_dict(torch.load('model_weights.pth'))
    # model_load.eval()#将模型设置为评估模式
    
    # #%% 保存和加载具有相同形状的模型
    # torch.save(model,'model.pth')
    # model_load_2 = torch.load('model.pth')
    
    #%% 绘制训练结果
    figure = plt.figure()
    plt.plot(loss_list)
    plt.show()
    figure = plt.figure()
    plt.plot(correct_list)
    plt.show()
