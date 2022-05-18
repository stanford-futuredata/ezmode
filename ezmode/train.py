import argparse
import sqlite3
import os
import time
import cv2
import numpy as np
import pandas as pd
import tqdm
import torch
import torchvision
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .data import DataLoader

class Trainer:
    def __init__(self, 
            model_backbone, 
            dataloader,
            model_path = None):

        self.model_backbone = model_backbone
        self.dataloader = dataloader
        self.model_path = model_path 

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def acc(self, labels, pred):

        acc = (labels == pred.argmax(-1)).float().mean().item()
        return acc

    def run(self, model, lr, train_data, nb_epochs):
        optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr,
                momentum=0.9,
                weight_decay=1e-4
                )

        for epoch in range(nb_epochs): 
            print("Training on Epoch={}".format(epoch+1))
            for i in tqdm.tqdm(range(len(train_data))):
                cur_row = train_data.iloc[i]
                img_path = cur_row['image']
                print(img_path)
                assert(os.path.exists(img_path))
                im = cv2.imread(img_path)
                inp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                inp = self.transform(inp)
                inp = inp.unsqueeze(0)

                boxes = torch.tensor([[cur_row['xmin'], cur_row['ymin'], cur_row['xmax'], cur_row['ymax']]]).type(torch.FloatTensor).cuda()
                labels = torch.tensor([cur_row['label']], dtype=torch.int64).cuda()
                image_id = torch.tensor([1], dtype=torch.int64).cuda()

                bbox_area =  (cur_row['xmax'] - cur_row['xmin']) * (cur_row['ymax'] - cur_row['ymin'])
                bbox_area = torch.tensor(bbox_area).cuda()

                iscrowd = torch.zeros((1,), dtype=torch.int64).cuda()


                target = {
                        'boxes': boxes, 
                        'labels': labels, 
                        'image_id': image_id, 
                        'area': bbox_area, 
                        'iscrowd': iscrowd
                        }
                inp = inp.cuda()
                model.cuda()
                outp, detections = model(inp, [target])
                loss_classifier = outp['loss_classifier']
                loss_box_reg = outp['loss_box_reg']
                loss_objectness = outp['loss_objectness']
                loss_rpn_box_reg = outp['loss_rpn_box_reg']

                print(outp)
                losses = sum(loss for loss in outp.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()


    def load_model(self, model_backbone, model_path, num_classes):

        print("Loading pre-trained weights for Faster-RCNN model...")
        if (model_backbone=='MobileNetV3'):
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained = True) 
            print("Loaded FRCNN weights with MobileNetV3 backbone!")

        if (model_backbone=='ResNet50'): 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True) 
            print("Loaded FRCNN weights with ResNet50 backbone!")

        if model_path is not None:
            print("Loading weights from {}...".format(model_path))
            model.load_state_dict(torch.load(model_path))

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.train()
        model.cuda()

        return model


    def save_model(self, 
            model, 
            lr, 
            nb_epochs):
        print("Saving trained Faster-RCNN model...")

        dest = os.path.join(self.dataloader.round_working_dir, 'model_lr={}_epochs={}_backbone={}.pth'.format(lr, nb_epochs, self.model_backbone))
        torch.save(model.state_dict(), dest)

        print("Done! Saved to {}".format(dest))
        return dest

    def train(self, lr, nb_epochs):

        train_data = self.dataloader.get_train_data()

        num_classes = self.dataloader.get_num_classes()
        print(num_classes)

        model = self.load_model(self.model_backbone, self.model_path, num_classes)

        self.run(model, lr, train_data, nb_epochs)  

        model_dest = self.save_model(model, lr, nb_epochs)

        return model_dest
