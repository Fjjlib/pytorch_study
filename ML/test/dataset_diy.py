# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:48:56 2024

@author: FJJ
"""

import os
import torch
import numpy as np
from torchvision.io import read_image


#%% 自定义数据集
# 必须实现三个函数：__init__、__len__、__getitem__
# class CustomImageDataset(Dataset):
#     # 
#     def __init__(self, annotation_file, img_dir, transform=None,target_transform=None):
#         self.img_labels = pd.read_csv(annotation_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
        
    
#     def __len__(self):
#         return len(self.img_labels)
    
#     # 加载并返回给定索引 idx 处的数据集样本
#     def __getitem__(self,idx):
#         img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
#         image = read_image(img_path)# 转化为张量
#         label = self.img_labels.iloc[idx,1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label # 元组形式
    
class GetLoader(torch.utils.data.Dataset):
    #initial
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
        
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label(index)
        return data, labels

    def __len__(self):
        return len(self.data)
    
    

device = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")
print(f'Using {device} device')

source_data = np.random.rand(10,20)
source_label = np.random.randint(0,2,(10,1))

torch_data = GetLoader(source_data, source_label)
datas = torch.utils.data.DataLoader(
    dataset=torch_data,
    batch_size=6,
    shuffle=True,
    drop_last = False,
    num_workers=1)

# 查看数据
for i,data in enumerate(datas):
    print("第 {} 个Batch \n{}".format(i, data))