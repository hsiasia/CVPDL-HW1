from transformers import DetrImageProcessor, DetrForObjectDetection

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import torchvision

import os
import json

import numpy as np
from PIL import Image, ImageDraw

from tqdm import tqdm

id2label = {0: "creatures", 1: "fish", 2: "jellyfish", 3: "penguin", 4: "puffin", 5: "shark", 6: "starfish", 7: "stingray"}

# The standard way in PyTorch to train a model is by creating datasets and a corresponding dataloaders. 
# Here we define a regular PyTorch dataset.
# Each item of the dataset is an image and corresponding annotations. 
class CocoDetection(torchvision.datasets.CocoDetection):
  def __init__(self, img_folder, processor):
    ann_file = os.path.join(img_folder, "_annotations.coco.json")
    # Torchvision already provides a CocoDetection dataset, which we can use
    super(CocoDetection, self).__init__(img_folder, ann_file)
    # DetrImageProcessor to resize + normalize the images and to turn the annotations (which are in COCO format) in the format that DETR expects. 
    # It will also resize the annotations accordingly.
    self.processor = processor

  def __getitem__(self, idx):
    # read in PIL image and target in COCO format
    # feel free to add data augmentation here before passing them to the next step
    img, target = super(CocoDetection, self).__getitem__(idx)

    # todo: data augmentation
    
    # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
    image_id = self.ids[idx]
    target = {'image_id': image_id, 'annotations': target}
    encoding = self.processor(images = img, annotations = target, return_tensors = "pt")
    pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
    target = encoding["labels"][0] # remove batch dimension

    return pixel_values, target

# dataloaders, which allow us to get batches of data. 
# We define a custom collate_fn to batch images together. 
# As DETR resizes images to have a min size of 800 and a max size of 1333, images can have different sizes. 
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  # pad images (pixel_values) to the largest image in a batch
  encoding = processor.pad(pixel_values, return_tensors = "pt")
  labels = [item[1] for item in batch]

  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  # pixel_mask to indicate which pixels are real (1)/which are padding (0)
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels

  return batch

# LightningModule, which is an nn.Module with some extra functionality.
# You can of course just train the model in native PyTorch as an alternative.
class Detr(pl.LightningModule):
  def __init__(self, lr, lr_backbone, weight_decay):
    super().__init__()
    # replace COCO classification head with custom head
    # we specify the "no_timm" variant here to not rely on the timm library
    # for the convolutional backbone
    self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                        revision="no_timm", 
                                                        num_labels = len(id2label),
                                                        ignore_mismatched_sizes = True)
    # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
    self.lr = lr
    self.lr_backbone = lr_backbone
    self.weight_decay = weight_decay

  def forward(self, pixel_values, pixel_mask):
    outputs = self.model(pixel_values = pixel_values, pixel_mask = pixel_mask)

    return outputs

  def common_step(self, batch, batch_idx):
    pixel_values = batch["pixel_values"]
    pixel_mask = batch["pixel_mask"]
    labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

    outputs = self.model(pixel_values = pixel_values, pixel_mask = pixel_mask, labels = labels)

    loss = outputs.loss
    loss_dict = outputs.loss_dict

    return loss, loss_dict

  def training_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)     
    # logs metrics for each training_step,
    # and the average across the epoch
    self.log("training_loss", loss)
    for k,v in loss_dict.items():
      self.log("train_" + k, v.item())

    return loss

  def validation_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)     
    self.log("validation_loss", loss)
    for k,v in loss_dict.items():
      self.log("validation_" + k, v.item())

    return loss

  def configure_optimizers(self):
    param_dicts = [
          {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
          {
            "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": self.lr_backbone,
          },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr = self.lr, weight_decay = self.weight_decay)

    return optimizer

  def train_dataloader(self):
    return train_dataloader

  def val_dataloader(self):
    return val_dataloader

if __name__ == '__main__':
  # train
  processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
  train_dataset = CocoDetection(img_folder = './dataset/train', processor = processor)
  val_dataset = CocoDetection(img_folder = './dataset/valid', processor = processor)

  train_dataloader = DataLoader(train_dataset, collate_fn = collate_fn, batch_size = 2, shuffle = True)
  val_dataloader = DataLoader(val_dataset, collate_fn = collate_fn, batch_size = 2)
  batch = next(iter(train_dataloader))

  # train
  model = Detr(lr=1e-4, lr_backbone=1e-4, weight_decay=1e-5)
  outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

  trainer = Trainer(accelerator='gpu', devices=1, max_steps=5000, gradient_clip_val=0.05, accumulate_grad_batches = 4)
  trainer.fit(model)

  torch.save(model.state_dict(), "models/model-DETR-best.ckpt")