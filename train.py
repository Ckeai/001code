from torchvision import datasets, transforms
import os
import random
from torch.utils.data import DataLoader
from model import *
import numpy as np
import torch
import torch.nn as nn



MAX_EPOCH=10
BATCH_SIZE=128
#数据集路径
#project_dir = os.path.dirname(os.path.abspath(__file__))
#train_dir = os.path.join(project_dir,"Mnist", "mnist_train")
#test_dir = os.path.join(project_dir,"Mnist", "mnist_test")

def set_seed(seed=1):
    random.seed(seed)#seed()方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数
    np.random.seed(seed)
    torch.manual_seed(seed)#设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同
    torch.cuda.manual_seed(seed)
set_seed()  # 设置随机种子

if torch.cuda.is_available():
    print("find gpu!")
else:
    raise AssertionError("not find GPU!!!")

#===============================1.数据预处理================================================
train_transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
])


test_transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),

])
#===============================2.数据输入====================================================
# 导入训练集
trainDataset = datasets.MNIST(root='./MNIST_data',transform=train_transform,train=True,download=True)
# 导入测试集
testDataset = datasets.MNIST(root='./MNIST_data',transform=test_transform,train=False,download=True)

train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE)

#==============================3.模型初始化=========================================================
net=AlexNet()
net = net.cuda()#模型放到gpu上
net.init()
# ==============================4.损失函数===================================
loss = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================= 5.优化器 ===============================
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================= 6.训练 ======================================
for epoch in range(MAX_EPOCH):
    loss_mean=0.
    losses = []
    correct = 0.
    total = 0.

    net.train()#
 #-----------------开始训练------------------
    for i, data in enumerate(train_loader):

       inputs, labels = data
       #数据放到GPU上
       device = torch.device(0)

       labels = labels.to(device)
       inputs = inputs.to(device)
       # forward
       predictions = net.forward(inputs)
       # backward

       Loss = loss(predictions, labels)
      # print(Loss)
       optimizer.zero_grad()

       print("=============更新之前===========")
       for name, parms in net.named_parameters():
           print(name, "is",parms.requires_grad)

       Loss.backward()
       # update weights
       optimizer.step()

       print("=============更新之后===========")
       for name, parms in net.named_parameters():
            print(name,"is",parms.requires_grad)

       losses.append(Loss.item())

       # 统计分类情况
       _, predicted = torch.max(predictions.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).squeeze().sum().cpu().detach().numpy()
       # 打印训练信息
       loss_mean += Loss.item()
       if (i + 1) % 10 == 0:
           loss_mean = loss_mean / 10
           print("Training:Epoch[{:0>3d}/{:0>3d}] Iteration[{:0>3d}/{:0>3d}] Loss: {:.4f} Acc:{:.2%}".format(
               epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
           loss_mean = 0.


# ============================= 7.测试======================================
loss_mean_test=0.
correct_test = 0.
total_test = 0.
net.eval()
with torch.no_grad():
    for j, data in enumerate(test_loader):
        test_inputs, test_labels = data
        # 数据放到GPU上
        device = torch.device(0)

        test_labels = test_labels.to(device)
        test_inputs = test_inputs.to(device)
        # forward
        test_predictions = net.forward(test_inputs)
        test_Loss = loss(test_predictions, test_labels)

    # 统计分类情况
        _, test_predicted = torch.max(test_predictions.data, 1)
        total_test += test_labels.size(0)
        correct_test += (test_predicted == test_labels).squeeze().sum().cpu().detach().numpy()
        # 打印训练信息
        loss_mean_test += test_Loss.item()
        if (j + 1) % 10 == 0:#每10个iteration打印一次
            loss_mean_test = loss_mean_test / 10
            print("Test:Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                j+ 1, len(test_loader), loss_mean_test, correct_test / total_test))
            loss_mean_test = 0.

