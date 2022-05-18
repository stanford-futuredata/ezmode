import torch
import os
import pandas as pd
import numpy as np
import argparse
import torchvision
import cv2

class FRCNNDataLoader(torch.utils.data.Dataset):
    def __init__(self, train_df):
        self.df = train_df
        self.xform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['image']
        assert(os.path.exists(fname))

        im = cv2.imread(fname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224), interpolation = cv2.INTER_AREA)
        inp = self.xform(im)

        boxes = torch.FloatTensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        labels = torch.tensor(row['label'], dtype=torch.int64)
        image_id = torch.tensor(idx, dtype=torch.int64)
        bbox_area = torch.tensor((row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']))

        target = {
                'boxes': boxes.cuda(),
                'labels': labels.cuda(),
                'image_id': image_id.cuda(),
                'area': bbox_area.cuda(),
        }
        return target, inp

