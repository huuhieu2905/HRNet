
import os
import random

import cv2
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import math
from config.defaults import _C as config
from utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class Face300W(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                self.landmarks_frame.iloc[idx, 0])
        # print(image_path)
        # scale = self.landmarks_frame.iloc[idx, 1]

        # center_w = self.landmarks_frame.iloc[idx, 2]
        # center_h = self.landmarks_frame.iloc[idx, 3]
        # center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        x1 = np.min(pts[:, 0]); x2 = np.max(pts[:, 0])
        y1 = np.min(pts[:, 1]); y2 = np.max(pts[:, 1])

        # center_w = (math.floor(x1) + math.ceil(x2)) / 2.0
        # center_h = (math.floor(y1) + math.ceil(y2)) / 2.0

        # scale = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1)) / 200.0
        # center = torch.Tensor([center_w, center_h])

        img = cv2.imread(image_path)
        # Convert the image to RGB (from BGR which is the default in OpenCV)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the image to a NumPy array with dtype float32
        img = np.array(img, dtype=np.float32)

        center = torch.Tensor([img.shape[1]//2, img.shape[0]/2])
        scale = max((img.shape[1]) / 256, (img.shape[0]) / 256)

        # scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]
            
            # pad_x = (x2 - x1) * 0.05
            # pad_y = (x2 - x1) * 0.05
            # img = img[int(x1):int(x2) + int(pad_x), int(y1):int(y2) + int(pad_y),:]
            # img = cv2.resize(img, [256, 256])
    
        # img = crop(img, center, scale, self.input_size, rot=r)
        img = img[int(y1):int(y2), int(x1):int(x2)]
        img = cv2.resize(img, (256, 256))
        new_img = img.copy()
        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                            scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return new_img, target, meta
            
            

if __name__ == '__main__':
    data = Face300W(config)
    img, target, meta = data.__getitem__(1410)
    for i in meta['tpts'].numpy():
        print(i)
        point = (int(i[0]), int(i[1]))
        cv2.circle(img, point, 1, (255,255,0), 1, cv2.LINE_AA)
    cv2.imwrite('img.jpg', img)
     