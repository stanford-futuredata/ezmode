import os
import numpy as np
import pandas as pd
import tqdm
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .data import DataLoader
from .utils import FRCNNDataLoader

class TrainEngine:
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

    def run(self, model, loader, lr, nb_epochs):
        model.train()
        optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = lr,
                momentum=0.9, weight_decay=1e-4
        )
        for epoch in tqdm.tqdm(range(nb_epochs)):
            self.train_epoch(model, optimizer, loader)

    def train_epoch(self, model, optimizer, loader):
        for batch_idx, (target, inp) in enumerate(loader):

            batched_target = []
            for i in range(len(target['labels'])):
                bbox = {}
                bbox['boxes'] = target['boxes'][i].reshape(1, -1)
                bbox['labels'] = torch.unsqueeze(target['labels'][i], dim = 0)
                bbox['image_id'] = torch.unsqueeze(target['image_id'][i], dim = 0)
                bbox['area'] = torch.unsqueeze(target['area'][i], dim = 0)
                batched_target.append(bbox)

            outp, detection = model(inp, batched_target)

            losses = sum(loss for loss in outp.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


    def load_model(self, model_backbone, model_path, num_classes):

        if (model_backbone=='MobileNetV3'):
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained = True) 
        elif (model_backbone=='ResNet50'): 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True) 
        if model_path is not None:
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

        dest = os.path.join(self.dataloader.round_working_dir, 'model_lr={}_epochs={}_backbone={}.pth'.format(lr, nb_epochs, self.model_backbone))
        torch.save(model.state_dict(), dest)

        print("Done Training! Model saved to {}".format(dest))
        return dest

    def train(self, lr, nb_epochs, batch_size):

        train_data = self.dataloader.get_train_data()
        dset = FRCNNDataLoader(train_data)
        loader = torch.utils.data.DataLoader(
                dataset = dset, 
                batch_size = batch_size, 
                shuffle = True)

        num_classes = self.dataloader.get_num_classes()
        model = self.load_model(self.model_backbone, self.model_path, num_classes)

        self.run(model, loader, lr, nb_epochs)  

        model_dest = self.save_model(model, lr, nb_epochs)
        return model_dest
