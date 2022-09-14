import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 测试集分辨率缩放至 256 x 256， 中心裁剪至 224 x 224，转换为Tensor格式，归一化均值为[0.485, 0.456, 0.406]， 标准差为[0.485, 0.456, 0.406]
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = "../0.jpg"
    img_path = "../1.jpg"
    # img_path = "../2.jpg"
    # img_path = "../3.jpg"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)  # 打开图片
    img = img.convert("RGB")  # 将图片格式转换为RGB
    plt.imshow(img)  # 显示图片
    # [N, C, H, W]
    img = data_transform(img)  # 数据预处理
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 扩充第一个维度（N,C,H,W）

    # read class_indict
    json_path = './class_indices.json'  # 类别标签路径
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # 加载类别标签
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=43).to(device)  # 实例化模型，并放置在device上

    # load model weights
    weights_path = "./resNet34.pth"  # 模型权重路径
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))  # 加载模型

    with torch.no_grad():  # 禁止计算模型权重梯度
        model.eval()

        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()  # 数据输入模型，并压缩数据维度
        predict = torch.softmax(output, dim=0)  # 通过softmax函数将模型输出转换为概率
        predict_cla = torch.argmax(predict).numpy()  # 计算概率最大值的索引

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],  # 输出预测类别
                                                 predict[predict_cla].numpy())  # 输出预测概率
    plt.title(print_res)  # 设置图片标题
    print(print_res)
    plt.show()  # 显示图片


if __name__ == '__main__':
    main()
