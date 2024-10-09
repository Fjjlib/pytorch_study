# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:33:16 2024

@author: FJJ
"""


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 搭建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            # 经过卷积层输出[16,28,28] 转入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 经过池化，输出[16,14,14]
        )
        
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            # 经过卷积输出[32,14,14]传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 经过池化输出[32,7,7] 传入输出层
            )
        
        # 输出层
        self.output = nn.Linear(in_features=32*7*7, out_features = 10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x) # batch  [32,7,7]
        x = x.view(x.size(0),-1) #保留batch 将后面的乘到一起[batch, 32*7*7]
        output = self.output(x)
        return output
    
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
        
        
if __name__=='__main__':   
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    #     print("Running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     print("Running on the CPU")
        
    # torch.cuda.device_count()
    device = torch.device("cpu")
    torch.manual_seed(1)# 为了每次实验结果一致
    # 设置超参数
    epoches = 10
    batch_size = 1000
    learning_rate = 0.001
    
    #训练集
    train_data = torchvision.datasets.MNIST(
        root = "./mnist/",
        train=True,
        transform = torchvision.transforms.ToTensor(),
        download=False,
        )
    
    # 显示训练集中的第一张
    print(train_data.train_data.size())
    plt.imshow(train_data.train_data[0].numpy())
    plt.show()
    
    #测试集
    test_data = torchvision.datasets.MNIST(
        root="./mnist/",
        train=False,
        )
    
    print(test_data.test_data.size())
    test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)/255
    test_y = test_data.test_labels
    
    #将训练集装入dataloader
    train_loader = Data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True, num_workers=3)
    
    #CNN 实例化
    cnn = CNN()
    print(cnn)
    
    #定义优化和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(),lr = learning_rate)
    loss_function = nn.CrossEntropyLoss()
    
    #开始训练
    accuracy_list = []
    for epoch in range(epoches):
        print(epoch)
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'\r{batch}',end='')
            
        test_output = cnn(test_x)
        pred_y = torch.max(test_output,1)[1].data.numpy()
        accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        print('Epoch: ',epoch, '| train loss: %.4f' % loss.data.numpy(),'|test accuracy:%.2f' % accuracy)
        accuracy_list.append(accuracy)
        
    plt.figure()
    plt.plot(accuracy_list)
    plt.show()