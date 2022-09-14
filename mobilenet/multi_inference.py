import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model_v2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path_list = ["../0.jpg", "../1.jpg",
                     "../3.jpg", "../2.jpg"]

    img_list = []
    for img_path in img_path_list:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = img.convert("RGB")

        img = data_transform(img)
        img_list.append(img)

    # batch img
    batch_img = torch.stack(img_list, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobileNetV2(num_classes=43).to(device)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        for idx, (pro, cla) in enumerate(zip(probs, classes)):
            print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
                                                             class_indict[str(cla.numpy())],
                                                             pro.numpy()))


if __name__ == '__main__':
    main()
