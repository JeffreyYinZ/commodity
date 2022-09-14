import os
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model_v2 import MobileNetV2
import sys
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append("..")
from commodity_dataset import MyDataSet
from data_tools import read_split_data, train_one_epoch, evaluate


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    tb_writer = SummaryWriter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    epochs = 100

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data("../commodity_dataset")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
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

    # create model
    net = MobileNetV2(num_classes=args.num_classes)

    model_weight_path = "./mobilenet_v2-pre.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # 删除分类器的权重
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # 冻结功能权重
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 构建一个优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    save_path = './MobileNetV2.pth'

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
        print("[epoch {}]  train_accuracy: {} val_accuracy: {}".format(epoch, round(train_acc, 4),
                                                                       round(val_acc, 4)))  # 打印测试精度
        tags = ["loss", "val_accuracy", "learning_rate", "train_accuracy"]  # tensorboard创建日志标签
        tb_writer.add_scalar(tags[0], mean_loss, epoch)  # 将训练平均损失加入训练日志
        tb_writer.add_scalar(tags[1], val_acc, epoch)  # 将测试精度加入训练日志
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)  # 将训练学习率加入训练日志
        tb_writer.add_scalar(tags[3], train_acc, epoch)  # 将训练学习率加入训练日志

        # 只保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=43)  # 类别数量
    parser.add_argument('--epochs', type=int, default=100)  # 训练周期
    parser.add_argument('--lr', type=float, default=0.001)  # 初始学习率
    parser.add_argument('--lrf', type=float, default=0.1)  # 学习率下降倍率

    opt = parser.parse_args()

    main(opt)
