from transformers import DetrImageProcessor, DetrForObjectDetection

import torch

import pytorch_lightning as pl

import os
from PIL import Image
from tqdm import tqdm
import json

import sys

folder = sys.argv[1]
output_path = sys.argv[2]
id2label = {0: "creatures", 1: "fish", 2: "jellyfish", 3: "penguin", 4: "puffin", 5: "shark", 6: "starfish", 7: "stingray"}

# LightningModule, which is an nn.Module with some extra functionality.
# You can of course just train the model in native PyTorch as an alternative.
class Detr(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm", 
                                                            num_labels = len(id2label),
                                                            ignore_mismatched_sizes = True
                                                            )

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values = pixel_values, pixel_mask = pixel_mask)
        return outputs

# predict test
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

model = Detr().to(device)
model.load_state_dict(torch.load("models/model-DETR-best.ckpt"))

allPreds = {}
files= os.listdir(folder)

for filename in tqdm(files):
    url = folder + filename
    image = Image.open(url)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():    
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0)[0]

        scoreList = []
        labelList = []
        boxList = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = score.to("cpu").to("cpu").numpy().tolist()
            label = label.to("cpu").detach().numpy().tolist()
            box = box.to("cpu").detach().numpy().tolist()

            scoreList.append(score)
            labelList.append(label)
            boxList.append(box)

            d = {}
            d['boxes'] = boxList
            d['labels'] = labelList
            d['scores'] = scoreList
            allPreds[filename] = d

with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(allPreds, file, ensure_ascii=False, indent=4)