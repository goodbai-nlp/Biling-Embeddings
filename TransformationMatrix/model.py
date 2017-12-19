#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: model.py
@time: 17-12-14 上午10:20
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

torch.cuda.set_device(2)

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, input):
        # Max pooling over a (2, 2) window
        input = F.max_pool2d(F.relu(self.conv1(input)),(2,2))
        # If the size is a square you can only specify a single number
        input = F.max_pool2d(F.relu(self.conv2(input)),2)
        input = input.view(-1,16*5*5)

        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = self.fc3(input)

        return input

    def num_flat_features(self,x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def data_init():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return(trainloader,testloader,classes)

# functions to show an image

def imshow(img):
    img = img/2+0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



if __name__ == "__main__":
    trainloader, testloader, classes = data_init()
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9)


    for epoch in range(2):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):

            inputs,labels = data
            inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda()) #把数据放到gpu中

            optimizer.zero_grad()

            #forward+backward+optimize
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss+=loss.data[0]
            if i%2000==1999:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
                running_loss=0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cuda().int() == labels.cuda().int()).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted.cuda().int() == labels.cuda().int()).squeeze()     # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
        for i in range(4):                      # 应该是len（c）=4
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))