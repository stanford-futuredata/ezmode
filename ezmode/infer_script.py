import pandas as pd
import datetime
import numpy as np
import argparse
import os
import tqdm
import torch
import torchvision
from collections import OrderedDict
from torchvision import transforms as trn
from torchvision.io import read_image
from torch.nn import functional as F
from PIL import Image
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import sqlalchemy
from ezmode import data

tx = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def set_gpu(gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    return 

def load_model(model_path, num_classes):
    print("Loading model...")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained = True) 

    if (model_path is not None):
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
        print("Loaded weights!")

    model.cuda()
    model.eval()
    return model


class ImageDSet(torch.utils.data.Dataset):
    def __init__(self, fnames, tx):
        self.fnames = fnames
        self.tx = tx

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        image = cv2.imread(img_path)
        image = Image.fromarray(image[:,:,::-1])
        image = self.tx(image)
        return image


def run_model(
        model,
        vid_base_path, 
        class_dict, 
        image_paths, 
        image_ids, 
        outp_file):

    dset = ImageDSet(image_paths, tx)
    batch_size = 1
    loader = torch.utils.data.DataLoader(
            dset, shuffle=False,
            batch_size=batch_size, num_workers=16
    )

    data_rows = []
    with torch.no_grad():
        for idx, batch  in enumerate(tqdm.tqdm(loader)):
            batch = batch.cuda(non_blocking=True)
            outps = model(batch)
            for i in range(len(outps)):

                cur_frame = idx*batch_size + i

                outp = outps[i]
                cur_boxes = outp['boxes'].cpu().detach().numpy()
                cur_scores = outp['scores'].cpu().detach().numpy()
                cur_labels = np.ndarray.tolist(outp['labels'].cpu().detach().numpy())

                cur_image_path = image_paths[cur_frame]
                cur_image_id = image_ids[cur_frame]

                num_outp = int(outp['labels'].shape[0])

                for outp_idx in range(num_outp):
                    data_rows.append([
                        vid_base_path, 
                        cur_frame,
                        cur_image_path, 
                        cur_image_id, 
                        cur_scores[outp_idx], 
                        class_dict[cur_labels[outp_idx]],
                        cur_boxes[outp_idx][0], 
                        cur_boxes[outp_idx][1], 
                        cur_boxes[outp_idx][2], 
                        cur_boxes[outp_idx][3]])

        logits = pd.DataFrame(
            data_rows, 
            columns= [
                'vid_base_path', 
                'frame', 
                'image_base_path',
                'image_id',
                'score',
                'label',
                'x1',
                'y1',
                'x2',
                'y2']
            )

        print(f'Writing inference results to {outp_file}')
        logits.to_csv(outp_file, index = False)
        print('Done!')
        return

def main():

    global args
    
    set_gpu(args.gpu)


    dataloader = data.DataLoader(
            project_name = args.project_name, 
            root = args.root, 
            working_dir = args.working_dir, 
            round_name = args.round_name,
            db = args.db)

    class_dict = dataloader.get_class_dict()
    val_data = pd.read_csv(dataloader.get_val_data())
    idxs = [i for i in range(len(val_data)) if ((i % 4) == args.gpu)]

    val_data = val_data.iloc[idxs]
    vid_base_paths = val_data['vid_base_path']

    model = load_model(
       model_path = args.model, 
       num_classes = args.num_classes)

    outp_dir = os.path.join(dataloader.round_working_dir, 'scores')

    if not os.path.isdir(outp_dir):
        os.mkdir(outp_dir)
    
    num_processed = len(os.listdir(outp_dir))
    print("{} PROCESSED OUT OF {}".format(num_processed, args.num_videos))



    for vid_base_path in vid_base_paths:

        outp_file = "{}.csv".format(vid_base_path.replace("/","_"))
        outp_file = os.path.join(outp_dir, outp_file)

        if (os.path.exists(outp_file)):
            continue

        print("Processing video: {}".format(vid_base_path))
        image_paths = dataloader.video_to_image_path(vid_base_path, root = True)
        image_ids = dataloader.video_to_id(vid_base_path)

        run_model(
                model, 
                vid_base_path, 
                class_dict, 
                image_paths, 
                image_ids, 
                outp_file)

    return



if __name__ == "__main__":
    global args 
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type = str)
    parser.add_argument('--model', type = str)
    parser.add_argument('--gpu', type = int)
    parser.add_argument('--project_name', type = str)
    parser.add_argument('--root', type = str)
    parser.add_argument('--working_dir', type = str)
    parser.add_argument('--round_name', type=str)
    parser.add_argument('--db', type = str)
    parser.add_argument('--num_classes', type = int)
    parser.add_argument('--num_videos', type = int)
    parser.add_argument('--to_process', type = int)
    args = parser.parse_args()
    main()
    
