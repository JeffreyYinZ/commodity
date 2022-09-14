import os
import argparse

import torch
from torchvision import transforms

import sys

sys.path.append("..")
from commodity_dataset import MyDataSet
from data_tools import read_split_data, train_one_epoch, evaluate
from model_v2 import MobileNetV2

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path,
                                                                                               val_rate=0.2)

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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    # create model
    net = MobileNetV2(num_classes=43).to(device)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--data-path', type=str,
                        default="../commodity_dataset")  ##################验证集路径

    parser.add_argument('--weights', type=str, default='./MobileNetV2',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
