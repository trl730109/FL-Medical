import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from image_processing import *
import os
from pathlib import Path

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
# from torchvision.io import read_image

import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

import pylab
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet50, resnet101
import torchvision.models.resnet
import numpy as np
from pathlib import Path


class MyDataset(Dataset):
    def __init__(self, root_dir, ann_txt_dir, transform=None):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        '''
        self.ann_txt_dir = ann_txt_dir
        self.root_dir = root_dir

        self.imgs_labels_dict = self.load_annotations()  # 返回字典，图片名为key，图片label为value
        # dataloader到时候会在这里取数据,必须是list类型
        self.imgs = [Path(os.path.join(self.root_dir, img)).as_posix() for img in list(self.imgs_labels_dict.keys())]  # 一个[包含图片路径]的list
        self.labels = [label for label in list(self.imgs_labels_dict.values())]  # [图像对应的labels] 的list
        print(self.imgs)
        # 相关预处理的初始化
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.transform = transform

    # __getitem__会执行 batch_size次，__getitem__返回的数据是给模型的
    def __getitem__(self, index): #图像和标签在当前list的索引，每次调用index是随机值，一个batch里的数据是随机打乱的
        image = self.imgs[index]
        label = self.labels[index]  # 得到单张图片和相应的标签（此处都是image都是文件目录）
        image = Image.open(image).convert('RGB')  # 得到图片数据
        # image = read_image(image)  # tensor类型

        if self.transform is not None:
            image = self.transform(image)  # 对图片进行某些变换
        label = torch.from_numpy(np.array(label))  # 转回Tensor格式
        # label = torch.tensor(label)  # 把图片标签也变成tensor类型
        return image, label

    def __len__(self):
        return len(self.imgs)

    def load_annotations(self):
        imgs_labels_dict = {}
        with open(self.ann_txt_dir) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]  # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            for filename, gt_label in samples:
                imgs_labels_dict[filename] = np.array(gt_label, dtype=np.int64)  # 将图片名和label组合成字典数据
        return imgs_labels_dict

# 把图片对应的tensor调整维度，并显示
def tensorToimg(img_tensor):
    img_tensor = img_tensor / 2 + 0.5  # 反标准化
    img = img_tensor.numpy()
    img = np.transpose(img, [1, 2, 0])  # 通道由[c,h,w]->[h,w,c]
    plt.imshow(img)
    # plt.imshow((img * 255).astype(np.uint8))
    plt.show()


def evaluate(model,loader):   #计算每次训练后的准确率
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)     #得到logits中分类值（要么是[1,0]要么是[0,1]表示分成两个类别）
        correct += torch.eq(pred,y).sum().float().item()        #用logits和标签label想比较得到分类正确的个数
    return correct/total


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # hospital_name = "D"
    # hospital_name = "F"
    # hospital_name = "G"
    hospital_name = "N"

    ann_txt_dir_train = './train_AP_'+hospital_name+'.txt'
    ann_txt_dir_val = './val_AP_'+hospital_name+'.txt'
    root_dir = os.getcwd()
    image_dir_name = 'Images_AP_all_'+hospital_name
    
    # image_path = Path(os.path.join(root_dir, image_dir_name)).as_posix()
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    results_save_folder = 'results_' + image_dir_name
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)

    batch_size = 8  #16
    num_epoch = 10
    model_weight_path = "./resnet50-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    num_class = 2
    lr = 0.01
    visualize = 'yes' # or 'no'

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     # transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(224),  # 长宽比例不变，将最小边缩放到256
                                   # transforms.CenterCrop(224),  # 再中心裁减一个224*224大小的图片
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}  # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_dataset = MyDataset(root_dir = root_dir, ann_txt_dir = ann_txt_dir_train, transform=data_transform["train"])
    train_num = len(train_dataset)
    print('train_num:',train_num)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=nw) #dataloader的标准输入

    validate_dataset = MyDataset(root_dir = root_dir, ann_txt_dir = ann_txt_dir_val, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size, shuffle=False,num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    if visualize == 'yes':
        # test one image-label pairs to see if correct in MyDataset --------------
        label_dic = {0: '0', 1: '1'}
        image, label = train_dataset.__getitem__(1)
        print('image.shape, label.numpy():', image.shape, label.numpy())
        print(label_dic[int(label.numpy())])
        tensorToimg(image)

        # test many image-label pairs to see if correct in MyDataset --------------
        dataiter = iter(train_loader)  # 迭代器
        images, labels = next(dataiter)  # 获取数据
        # 显示图像
        tensorToimg(torchvision.utils.make_grid(images))
        print('labels.numpy():', labels.numpy())

    # 若不使用迁移学习的方法，注释掉61-69行，并net = resnet34(num_calsses参数)
    # net = resnet34()    # 未传入参数，最后一个全连接层有1000个结点
    net = resnet50()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    #  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    #  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    #  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    # net.load_state_dict载入模型权重。torch.load(model_weight_path)载入到内存当中还未载入到模型当中
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

    # 冻结部分权重
    for param in net.parameters():  #对于模型的每个权重，使其不进行反向传播，即固定参数
        param.requires_grad = False
    for param in net.fc.parameters(): # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
        param.requires_grad = True

    # change fc layer structure
    in_channel = net.fc.in_features # 输入特征矩阵的深度。net.fc是所定义网络的全连接层
    net.fc = nn.Linear(in_channel, num_class)  # 类别个数
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    # optimizer settings
    # optimizer = optim.Adam(net.parameters(), lr=lr) #会报错，因为optimizer的输入参数parameters必须都是可以修改、反向传播的，即requires_grad=True,但是我们刚才已经固定了除了最后一层的所有参数，所以会出错。
    # filter()函数过滤掉parameters中requires_grad=Fasle的参数, #重要的是这一句
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)  #大概率比SGD收敛的好！！！！！！

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    #mode:'min’模式检测metric是否不再减小，'max’模式检测metric是否不再增大; factor 学习率每次降低多少, patience,容忍网路的性能不提升的次数,高于这个次数就降低学习率

    best_acc = 0.0
    for epoch in range(num_epoch):
        # train
        net.train()
        train_loss = 0.0
        train_acc = 0.
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad() #优化器先清零，不然会叠加上次的数值
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            pred = torch.max(logits, 1)[1]

            # print(pred)------
            train_correct = torch.max(logits, dim=1)[1]
            # train_correct = (pred == labels).sum()
            train_acc += (train_correct == labels.to(device)).sum().item()
            # train_acc += train_correct.item()

            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            # print train process
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")

        # save training results
        strr=str(epoch)+"  "+str(train_loss/len(train_dataset)*100)+'    '+str(train_acc/len(train_dataset)*100)
        with open(Path(os.path.join(results_save_folder, 'train_loss.txt')).as_posix(), 'a') as f:
            f.write(str(strr) + '\n')

        # validate
        net.eval() # 控制训练过程中的Batch normalization
        eval_acc = 0.0  # accumulate accurate number / epoch
        eval_loss = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                loss = loss_function(outputs, val_labels)
                eval_loss += loss.item()

                predict_y = torch.max(outputs, dim=1)[1]
                eval_acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = eval_acc / val_num
            if val_accurate >= best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), Path(os.path.join(results_save_folder, 'best_resNet50.pth')).as_posix())
            print('[epoch %d] val_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, eval_loss / step, val_accurate))

            # save val results
            with open(Path(os.path.join(results_save_folder, 'val_accurate.txt')).as_posix(), 'a') as f:
                f.write(str(val_accurate) + '\n')

        with open(Path(os.path.join(results_save_folder, 'best_acc.txt')).as_posix(), 'a') as f:
            f.write(str(best_acc) + '\n')

        # scheduler的优化是在每一轮后面进行的
        scheduler.step(eval_loss)

        # print('epoch: ', epoch, 'lr: ', scheduler.get_lr())
        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]["lr"])

    print('Finished Training')