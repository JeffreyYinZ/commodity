import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import resnet34
import torch.nn as nn

import sys

sys.path.append("..")
from commodity_dataset import MyDataSet
from data_tools import read_split_data, train_one_epoch, evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 设置运算设备，首选gpu

    print(args)  # 打印自定义变量
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()  # 创建tensorboard SummaryWriter实例，用作记录模型训练信息

    # 获取训练集路径，训练集标签， 测试集路径，测试集标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 创建数据预处理操作实例；训练集随机裁剪致224x224分辨率，随机水平翻转，增加数据集，转换为Tensor格式；归一化均值为[0.485, 0.456, 0.406]， 标准差为[0.485, 0.456, 0.406]
    # 测试集分辨率缩放至 256 x 256， 中心裁剪至 224 x 224，转换为Tensor格式，归一化均值为[0.485, 0.456, 0.406]， 标准差为[0.485, 0.456, 0.406]
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 创建训练集和测试集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    # 设置训练batch_size
    batch_size = args.batch_size
    # 创建加载数据的进程数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])
    print('Using {} dataloader workers every process'.format(nw))

    # 加载训练集和测试集
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    net = resnet34()  # 创建网络模型
    model_weight_path = args.weights  # 预训练权重路径
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载预训练模型至模型

    in_channel = net.fc.in_features  # 获取模型最后一层的输入维度
    net.fc = nn.Linear(in_channel, args.num_classes)  # 将模型最后一层输出维度设置为 数据集分类数量（3）
    net.to(device)  # 将模型加载至设备（首选gpu）

    # 创建优化器（随机梯度下降法）， 设置学习率， 动量， 权重衰减系数
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    # 创建一个余弦函数来更新学习率，训练完成后，学习率是开始时的0.1倍。
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf

    # 实例化lr_scheduler方法，用于更新学习率
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0
    for epoch in range(args.epochs):
        # 训练一个周期
        mean_loss, train_sum_num = train_one_epoch(model=net,
                                                   optimizer=optimizer,
                                                   data_loader=train_loader,
                                                   device=device,
                                                   epoch=epoch)

        scheduler.step()  # 更新学习率

        # 测试算法精度
        sum_num = evaluate(model=net,
                           data_loader=val_loader,
                           device=device)
        val_acc = sum_num / len(val_data_set)  # 测试集中正确预测数量/测试集总数量，得出测试精度
        train_acc = train_sum_num / len(train_data_set)
        print("[epoch {}]  train_accuracy: {} val_accuracy: {}".format(epoch, round(train_acc, 4), round(val_acc, 4)))  # 打印测试精度
        tags = ["loss", "val_accuracy", "learning_rate", "train_accuracy"]  # tensorboard创建日志标签
        tb_writer.add_scalar(tags[0], mean_loss, epoch)  # 将训练平均损失加入训练日志
        tb_writer.add_scalar(tags[1], val_acc, epoch)  # 将测试精度加入训练日志
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)  # 将训练学习率加入训练日志
        tb_writer.add_scalar(tags[3], train_acc, epoch)  # 将训练学习率加入训练日志

        # 只保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), "./resNet34.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=43)  # Number of categories
    parser.add_argument('--epochs', type=int, default=20)  # Training cycles
    parser.add_argument('--batch-size', type=int, default=16)  # batch_size
    parser.add_argument('--lr', type=float, default=0.001)  # Initial learning rate
    parser.add_argument('--lrf', type=float, default=0.1)  # Learning rate decline multiplier
    parser.add_argument('--data-path', type=str,
                        default="../commodity_dataset")  # 数据集路径
    parser.add_argument('--weights', type=str, default='./resnet34-pre.pth',  # 预训练模型路径
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')  # 训练设备

    opt = parser.parse_args()

    main(opt)
