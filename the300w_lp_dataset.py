import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from augment import augment_data
from utils.sys_utils import the300w_lp_bbox, euler_from_file, quat_from_file, vector_from_file, lie_from_file


class The300WLPDataset(Dataset):
    def __init__(self, opts, is_train):
        super().__init__()
        self.opts = opts
        # self.train_list_dict = {"300W-LP": "300w_lp_train.txt", 
        #                         "BIWI": "biwi_train.txt"}
        # self.test_list_dict = {"300W-LP": "300w_lp_test.txt", 
        #                        "AFLW2000": "300w_lp_test.txt", 
        #                         "BIWI": "biwi_all.txt"}
        if is_train:
            self.dataset_path = self.opts.train_dataset_path
            # self.img_list_filename = self.train_list_dict[self.opts.train_dataset]
        else:
            self.dataset_path = self.opts.val_dataset_path
            # self.img_list_filename = self.test_list_dict[self.opts.val_dataset]

        self.img_list = []
        self.label_list = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.img_list.append(os.path.join(root, file))
                if file.endswith('.txt'):
                    self.label_list.append(os.path.join(root, file))
        self.img_list = sorted(self.img_list)
        self.label_list = sorted(self.label_list)
        self.img_size = self.opts.img_size
        self.rot_type = self.opts.rot_type
        self.is_train = is_train

        # with open(self.img_list_filename, "r") as f:
        #     fts = f.read().splitlines()
        #     self.img_list = [os.path.join(self.dataset_path, "imgs", x) for x in fts]
        #     self.label_list = [x.replace("imgs/", "labels/").replace(".jpg", ".txt") for x in self.img_list]

        self.length = len(self.img_list)


    def __getitem__(self, index):
        
        # read image and augment
        img = cv2.imread(self.img_list[index])
        # print(index)
        x_min, y_min, x_max, y_max = the300w_lp_bbox(self.label_list[index])
        img = img[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        if self.is_train:
            img = augment_data(img)
        else:
            img = np.transpose(img, axes=(2, 0, 1)) # (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img) / 255. # to range [0, 1]

        # https://pytorch.org/vision/stable/transforms.html
        # Change RGB to BGR
        normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        img = normalize(img)

        # read label
        # ["rot_mat", "quat", "lie", "euler"]
        eulers = torch.tensor(euler_from_file(self.label_list[index]))
        if self.rot_type == "euler":
            eulers = euler_from_file(self.label_list[index])
            label = torch.tensor(eulers)
        elif self.rot_type == "quat":
            quat = quat_from_file(self.label_list[index])
            label = torch.tensor(quat)
        elif self.rot_type == "rot_mat":
            l_vec, d_vec, f_vec = vector_from_file(self.label_list[index])
            label = torch.tensor([l_vec, d_vec, f_vec]).flatten()
        elif self.rot_type == "lie":
            lies = lie_from_file(self.label_list[index])
            label = torch.tensor(lies)
        return [img, label], [self.img_list[index], self.label_list[index], eulers]


    def __len__(self):
        return self.length


class NPZ_Dataset(Dataset):
    def __init__(self, opts, is_train) -> None:
        super().__init__()
        self.opts = opts
        if is_train:
            self.dataset_path = self.opts.train_dataset_path
        else:
            self.dataset_path = self.opts.val_dataset_path

        dataset = np.load(self.dataset_path)
        self.img_list, self.pose_list = dataset["image"], dataset["pose"]
        self.rot_type = self.opts.rot_type
        self.is_train = is_train

        self.length = len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        eulers = self.pose_list[index]

    def __len__(self):
        return self.length
        