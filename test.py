import torchvision
from torchvision import models

import torch
import torch.nn as nn

import os
import cv2
from tqdm import tqdm
import json

import sys

folder = sys.argv[1]
output_path = sys.argv[2]

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

class ObjectDetect(nn.Module):
    def __init__(self):
        super(ObjectDetect, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    def forward(self, input, target = None):
        output = self.model(input, target)
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# predict test
files= os.listdir(folder)

model = ObjectDetect().to(device)
model.load_state_dict(torch.load('./model-FRCNN-best.ckpt'))
model.eval() # Set your model to evaluation mode.
transforms_test = get_transform()

allPreds = {}
for filename in tqdm(files):
    url = folder + filename
    image = cv2.imread(url)
    image = transforms_test(image)

    imgsList = [image.to(device)]
    with torch.no_grad():                   
        pred = model(imgsList)
        for i in range(len(pred)):
            d = {}
            d['boxes'] = pred[i]['boxes'].to("cpu").numpy().tolist()
            # allBox = []
            # for box in d['boxes']:
            #     xmin = box[0]
            #     ymin = box[1]
            #     w = box[2] - xmin
            #     h = box[3] - ymin
            #     allBox.append([xmin, ymin, w, h])
            # d['boxes'] = allBox
            d['labels'] = pred[i]['labels'].to("cpu").numpy().tolist()
            d['scores'] = pred[i]['scores'].to("cpu").numpy().tolist()
        allPreds[filename] = d
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(allPreds, file, ensure_ascii=False, indent=4)