# Regression
# Numerical Operations
import math
import numpy as np
# IO
# import pandas as pd
import os
# import csv
# For Progress Bar
from tqdm import tqdm
# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# For plotting learning curve
# from torch.utils.tensorboard import SummaryWriter

# CNN
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
# from torch.utils.data import ConcatDataset, Subset
import torchvision.transforms as transforms
# from torchvision.datasets import DatasetFolder, VisionDataset
from PIL import Image
import random

# Mine
import json
import torchvision
from torchvision import models
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pycocotools.coco import COCO

# from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 75029,  # Your seed number, you can pick your lucky number. :)
    'valid_ratio': 0.2, # validation_size = train_size * valid_ratio
    'n_epochs': 100, # Number of epochs.            
    'batch_size': 1,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stop': 25,   # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model-',  # Your model will be saved here.
    'max_len': 64
}

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5), 
    A.ColorJitter(p=0.2),
    # ToTensorV2(),
],
    A.BboxParams(format='coco', label_fields=['category_ids'])
)

class ODDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation = None, transforms = None, train = True):
        self.root = root

        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.transforms = transforms
        self.train = train

    def __getitem__(self, index):
        try:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotation = self.coco.loadAnns(ann_ids)

            img_path = self.coco.loadImgs(img_id)[0]['file_name']
            img = cv2.imread(os.path.join(self.root, img_path))

            num_objs = len(annotation)
            bboxes = []
            labels = []
            areas = []
            iscrowd = []
            for i in range(num_objs):
                bboxes.append(annotation[i]['bbox'])
                labels.append(annotation[i]['category_id'])
                areas.append(annotation[i]['area'])
                iscrowd.append(annotation[i]['iscrowd'])

            if self.train is True:
                augmented = aug(image=img, bboxes=bboxes, category_ids=labels)
                img = augmented['image']
                bboxes = augmented['bboxes']

            mod_bboxes = []
            for i in range(num_objs):
                xmin = bboxes[i][0]
                ymin = bboxes[i][1]
                xmax = xmin + bboxes[i][2]
                ymax = ymin + bboxes[i][3]
                mod_bboxes.append([xmin, ymin, xmax, ymax])

            mod_bboxes = mod_bboxes[:len(mod_bboxes)] + [[1, 1, 2, 2]] * max(0, config['max_len'] - len(mod_bboxes))
            labels = labels[:len(labels)] + [-1] * max(0, config['max_len'] - len(labels))
            areas = areas[:len(areas)] + [1] * max(0, config['max_len'] - len(areas))
            iscrowd = iscrowd[:len(iscrowd)] + [0] * max(0, config['max_len'] - len(iscrowd))
        except:
            return img, None

        mod_bboxes = torch.as_tensor(mod_bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = mod_bboxes
        my_annotation["labels"] = labels
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        my_annotation["img_name"] = img_path

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation
    
    def __len__(self):
        return len(self.ids)
    
def collate_fn(batch):
    return tuple(zip(*batch))

class ObjectDetect(nn.Module):
    def __init__(self):
        super(ObjectDetect, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    def forward(self, input, target = None):
        output = self.model(input, target)
        return output

if __name__ == '__main__':
    same_seed(config['seed'])

    train_data_dir = "./dataset/train"
    valid_data_dir = "./dataset/valid"
    train_coco = "./dataset/train/_annotations.coco.json"
    valid_coco = "./dataset/valid/_annotations.coco.json"

    train_set = ODDataset(train_data_dir, train_coco, get_transform(), True)
    train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle = True, num_workers = 0, pin_memory = True, collate_fn=collate_fn)
    # print(train_set[0])
    # print(len(train_set))
    # for batch in tqdm(train_loader):
    #     imgs, targets = batch
    #     print(imgs)
    #     print(targets)
    #     exit()
    # exit()

    valid_set = ODDataset(valid_data_dir, valid_coco, get_transform(), False)
    valid_loader = DataLoader(valid_set, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True, collate_fn=collate_fn)
    # print(valid_set[0])
    # print(len(valid_set))
    # for batch in tqdm(valid_loader):
    #     imgs, targets = batch
    #     print(imgs)
    #     print(targets)
    #     exit()
    # exit()

    # Initialize a model, and put it on the device specified.
    model = ObjectDetect().to(device)
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.SGD(model.parameters(), lr = config['learning_rate'], momentum = 0.9, weight_decay = config['weight_decay'])

    n_epochs, best_loss, best_reg_loss, step, early_stop_count = config['n_epochs'], math.inf, math.inf, 0, 0

    for epoch in range(n_epochs):
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_cls_loss = []
        train_reg_loss = []
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, targets = batch
            imgsList = list(image.to(device) for image in imgs)
            targetsList = [] 
            for i in range(len(imgsList)):
                d = {}
                d['boxes'] = targets[i]['boxes'].to(device)
                d['labels'] = targets[i]['labels'].to(device)
                targetsList.append(d)

            # Forward the data. (Make sure data and model are on the same device)
            loss_dict = model(imgsList, targetsList)
            # output: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
            total_loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()   # Set gradient to zero.
            total_loss.backward()   # Compute gradient(backpropagation).
            optimizer.step()    # Update parameters.
            
            step += 1

            train_loss.append(total_loss.detach().item())
            train_cls_loss.append(loss_dict["loss_classifier"].detach().item())
            train_reg_loss.append(loss_dict["loss_box_reg"].detach().item())

        mean_train_loss = sum(train_loss)/len(train_loss)
        mean_train_cls_loss = sum(train_cls_loss)/len(train_cls_loss)
        mean_train_reg_loss = sum(train_reg_loss)/len(train_reg_loss)

        # model.eval()
        valid_loss = []
        valid_cls_loss = []
        valid_reg_loss = []
        for batch in tqdm(valid_loader):
            imgs, targets = batch
            imgsList = list(image.to(device) for image in imgs)
            targetsList = [] 
            for i in range(len(imgsList)):
                d = {}
                d['boxes'] = targets[i]['boxes'].to(device)
                d['labels'] = targets[i]['labels'].to(device)
                targetsList.append(d)
            loss_dict = model(imgsList, targetsList)
            total_loss = sum(loss for loss in loss_dict.values())

            valid_loss.append(total_loss.detach().item())
            valid_cls_loss.append(loss_dict["loss_classifier"].detach().item())
            valid_reg_loss.append(loss_dict["loss_box_reg"].detach().item())

        mean_valid_loss = sum(valid_loss)/len(valid_loss)
        mean_valid_cls_loss = sum(valid_cls_loss)/len(valid_cls_loss)
        mean_valid_reg_loss = sum(valid_reg_loss)/len(valid_reg_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}]: Total train loss: {mean_train_loss:.4f}, Cls train loss: {mean_train_cls_loss:.4f}, Reg train loss: {mean_train_reg_loss:.4f}')
        print(f'Epoch [{epoch+1}/{n_epochs}]: Total valid loss: {mean_valid_loss:.4f}, Cls valid loss: {mean_valid_cls_loss:.4f}, Reg valid loss: {mean_valid_reg_loss:.4f}')
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'] + "FRCNN-best.ckpt") # Save your best model
            print('Saving model with best loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            torch.save(model.state_dict(), config['save_path'] + str(epoch) + ".ckpt") # Save your best model
            print('Saving model with loss {:.3f}...'.format(mean_valid_loss))
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break
    # exit()


    # # predict valid
    # model = ObjectDetect().to(device)
    # model.load_state_dict(torch.load(config['save_path'] + "best.ckpt"))
    # model.eval() # Set your model to evaluation mode.

    # allPreds = {}
    # for batch in tqdm(valid_loader):
    #     imgs, targets = batch
    #     imgsList = list(image.to(device) for image in imgs)  
    #     with torch.no_grad():                   
    #         pred = model(imgsList)
    #         for i in range(len(pred)):
    #             d = {}
    #             d['boxes'] = pred[i]['boxes'].to("cpu").numpy().tolist()
    #             # allBox = []
    #             # for box in d['boxes']:
    #             #     xmin = box[0]
    #             #     ymin = box[1]
    #             #     w = box[2] - xmin
    #             #     h = box[3] - ymin
    #             #     allBox.append([xmin, ymin, w, h])
    #             # d['boxes'] = allBox
    #             d['labels'] = pred[i]['labels'].to("cpu").numpy().tolist()
    #             d['scores'] = pred[i]['scores'].to("cpu").numpy().tolist()
    #         allPreds[targets[0]["img_name"]] = d
    # with open('outputs/output-FRCNN-valid.json', 'w', encoding='utf-8') as file:
    #     json.dump(allPreds, file, ensure_ascii=False, indent=4)


    # # predict test
    # folder = "./dataset/test/"
    # files= os.listdir("./dataset/test")

    # model = ObjectDetect().to(device)
    # model.load_state_dict(torch.load(config['save_path'] + "best.ckpt"))
    # model.eval() # Set your model to evaluation mode.
    # transforms_test = get_transform()

    # allPreds = {}
    # for filename in tqdm(files):
    #     url = folder + filename
    #     image = cv2.imread(url)
    #     image = transforms_test(image)

    #     imgsList = [image.to(device)]
    #     with torch.no_grad():                   
    #         pred = model(imgsList)
    #         for i in range(len(pred)):
    #             d = {}
    #             d['boxes'] = pred[i]['boxes'].to("cpu").numpy().tolist()
    #             # allBox = []
    #             # for box in d['boxes']:
    #             #     xmin = box[0]
    #             #     ymin = box[1]
    #             #     w = box[2] - xmin
    #             #     h = box[3] - ymin
    #             #     allBox.append([xmin, ymin, w, h])
    #             # d['boxes'] = allBox
    #             d['labels'] = pred[i]['labels'].to("cpu").numpy().tolist()
    #             d['scores'] = pred[i]['scores'].to("cpu").numpy().tolist()
    #         allPreds[filename] = d
    # with open('outputs/output-FRCNN-test.json', 'w', encoding='utf-8') as file:
    #     json.dump(allPreds, file, ensure_ascii=False, indent=4)