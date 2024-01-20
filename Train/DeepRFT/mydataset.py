import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random


def is_require_npy(filename, p_set):
    if filename.endswith('npy'):
        name = filename.split('-')
        return name[1] in p_set
    else:
        return False

class DataLoaderTrain(Dataset):
    def __init__(self, input_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.img_options = img_options
        self.ps = self.img_options['patch_size']
        self.num = self.img_options['train_size']

        p_set = set()
        for i in range(self.num):
            if i < 90:
                i = i + 1
            else:
                i = i + 1 + 5
            p_set.add('P' + str(i).zfill(3))
        AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10', 'FullSample']
        BFtype = ['AccFactor04-2', 'AccFactor08-2', 'AccFactor10-2', 'FullSample-2']
        inp_files = list()
        tar_file = list()
        for a in range(len(AFtype) - 1):
            inp_dirs = sorted(os.listdir (os.path.join(input_dir, AFtype[a])))
            inp_files.extend([os.path.join(input_dir, AFtype[a], x) for x in inp_dirs if is_require_npy(x, p_set)])
            tar_dirs = sorted(os.listdir (os.path.join(input_dir, AFtype[3])))
            tar_file.extend([os.path.join(input_dir, AFtype[3], x) for x in tar_dirs if is_require_npy(x, p_set)])
            inp_dirs = sorted(os.listdir (os.path.join(input_dir, BFtype[a])))
            inp_files.extend([os.path.join(input_dir, BFtype[a], x) for x in inp_dirs if is_require_npy(x, p_set)])
            tar_dirs = sorted(os.listdir (os.path.join(input_dir, BFtype[3])))
            tar_file.extend([os.path.join(input_dir, BFtype[3], x) for x in tar_dirs if is_require_npy(x, p_set)])

        self.inp_filenames = inp_files
        self.tar_filenames = tar_file

        self.size = len(self.tar_filenames)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index_ = index % self.size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_array = np.load(inp_path)
        inp_img = torch.tensor(np.expand_dims(inp_array, axis=0), dtype=torch.float32)
        tar_array = np.load(tar_path)
        tar_img = torch.tensor(np.expand_dims(tar_array, axis=0), dtype=torch.float32)

        w, h = tar_array.shape  # 512 * 246, 448 * 132 等

        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = F.pad(inp_img, [0, 0, padh, padw], padding_mode='reflect')
            tar_img = F.pad(tar_img, [0, 0, padh, padw], padding_mode='reflect')

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)
        
        
        # gamma correction
        ga = random.randint(0, 2)
        gamma = np.random.uniform(0.5, 2.0)
        if ga == 1:
            inp_img = F.adjust_gamma(inp_img, gamma)
            tar_img = F.adjust_gamma(tar_img, gamma)
        
        

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, input_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        self.img_options = img_options
        self.ps = self.img_options['patch_size']
        self.num = self.img_options['train_size']

        p_set = set()
        for i in range(self.num):
            i = i + 90 + 1
            p_set.add('P' + str(i).zfill(3))
        AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10', 'FullSample']
        inp_files = list()
        tar_file = list()
        for a in range(len(AFtype) - 1):
            inp_dirs = sorted(os.listdir (os.path.join(input_dir, AFtype[a])))
            inp_files.extend([os.path.join(input_dir, AFtype[a], x) for x in inp_dirs if is_require_npy(x, p_set)])
            tar_dirs = sorted(os.listdir (os.path.join(input_dir, AFtype[3])))
            tar_file.extend([os.path.join(input_dir, AFtype[3], x) for x in tar_dirs if is_require_npy(x, p_set)])

        self.inp_filenames = inp_files
        self.tar_filenames = tar_file

        self.size = len(self.tar_filenames)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index_ = index % self.size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_array = np.load(inp_path)
        inp_img = torch.tensor(np.expand_dims(inp_array, axis=0), dtype=torch.float32)
        tar_array = np.load(tar_path)
        tar_img = torch.tensor(np.expand_dims(tar_array, axis=0), dtype=torch.float32)

        w, h = tar_array.shape  # 512 * 246, 448 * 132 等

        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = F.pad(inp_img, [0, 0, padh, padw], padding_mode='reflect')
            tar_img = F.pad(tar_img, [0, 0, padh, padw], padding_mode='reflect')

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)

        # Crop patch
        inp_img = inp_img[:, (hh - ps) // 2: (hh + ps) // 2, (ww - ps) // 2: (ww + ps) // 2]
        tar_img = tar_img[:, (hh - ps) // 2: (hh + ps) // 2, (ww - ps) // 2: (ww + ps) // 2]

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return tar_img, inp_img, filename