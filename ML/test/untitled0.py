# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:19:31 2024

@author: FJJ
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# 读取小批量数据样本
def data_iter(batch_size,features,labels):
    '''
    

    Parameters
    ----------
    batch_size : int
        number of samples per training
        
    features : array(n,m)
        training data
    labels : array(n,1)
        the label of training data

    Returns
    -------
    None.

    '''
    
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples, batch_size):
        j = np.array(indices[i:min(i+batch_size,num_examples)])
        yield features.take(j), labels.take(j)
        
if __name__ == '__main__':
    num_inputs = 2
    num_examples = 1000
    true_w = [2,-3.4]
    true_b = 4.2
    features = np.random.normal(scale = 1,size=(num_examples,num_inputs))
    labels = true_w[0]*features[:,0] + true_w[1] * features[:,1] +true_b;
    
    #添加高斯噪声
    labels += np.random.normal(scale=0.01,size = labels.shape)
    
    plt.figure()
    plt.scatter(features[:,1],labels,s=1,c='red')
    plt.title('标签与特征2的关系')
    plt.xlabel('特征2')
    plt.ylabel('标签')
    
    # 读取小批量数据
    batch_size = 10
    for X,y in data_iter(batch_size,features,labels):
        print(X,y)
        break
    
    # 模型权重初始化
    w = np.random.normal(scale=0.01,size=(num_inputs,1))
    b = np.zeros(shape=(1,))
    
    # 创建梯度
    w.attach_grad()
    b.attach_grad()
    
    