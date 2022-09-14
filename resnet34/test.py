import os
import argparse

import torch
from torchvision import transforms
from model import resnet34

import sys

sys.path.append("..")
from commodity_dataset import MyDataSet
from data_tools import read_split_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# 此文件注释可参考 train_resnet.py


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    # create model
    net = resnet34(num_classes=args.num_classes).to(device)

    # load model weights
    weights_path = "resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device), strict=True)

    ############################################################################
    correct = list(0. for i in range(args.num_classes))
    total = list(0. for i in range(args.num_classes))
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(val_loader):
            images, labels = data

            output = net(images.to(device))

            prediction = torch.argmax(output, 1)
            res = prediction == labels.to(device)
            for label_idx in range(len(labels)):
                label_single = labels[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1
        acc_str = 'Accuracy: %.4f' % (sum(correct) / sum(total))
        for acc_idx in range(args.num_classes):
            try:
                acc = correct[acc_idx] / total[acc_idx]
            except:
                acc = 0
            finally:
                acc_str += '\tclassID:%d\tacc:%.4f\t' % (acc_idx + 1, acc)
        print(acc_str)


############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=43)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--data-path', type=str,
                        default="../commodity_dataset")  ###################验证集单独路径，和元数据集格式相同！

    parser.add_argument('--weights', type=str, default='./resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
