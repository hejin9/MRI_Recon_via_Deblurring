import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random

def p_set_list_train(num):
    p_set = set()
    if num > 90:
        for i in range(90):
            i = i + 1
            p_set.add('P' + str(i).zfill(3))
        for j in range(num - 90):
            j = j + 95 + 1
            p_set.add('P' + str(j).zfill(3))
    else:
        for i in range(num):
            i = i + 1
            p_set.add('P' + str(i).zfill(3))
    return p_set

def p_set_list_val(num):
    p_set = set()
    for j in range(num):
        j = j + 90 + 1
        p_set.add('P' + str(j).zfill(3))
    return p_set


def is_require_npy(filename, p_set):
            
    if filename.endswith('npy'):
        name = filename.split('-')
        return name[1] in p_set
    else:
        return False


class DataLoaderTrain(Dataset):
    def __init__(self, input_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.frame = img_options['frame']
        num = img_options['train_size']
        
        inp_files = list()
        tar_files = list()
        acc_AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10']
        ful_AFtype = ['FulSample04', 'FulSample08', 'FulSample10']
        p_set = p_set_list_train(num)
        for a in range(len(acc_AFtype)):
        # for a in range(1):
            inp_dirs = sorted(os.listdir(os.path.join(input_dir, acc_AFtype[a])))
            inp_files.extend([os.path.join(input_dir, acc_AFtype[a], x) for x in inp_dirs if is_require_npy(x, p_set)])
            tar_dirs = sorted(os.listdir(os.path.join(input_dir, ful_AFtype[a])))
            tar_files.extend([os.path.join(input_dir, ful_AFtype[a], x) for x in tar_dirs if is_require_npy(x, p_set)])
        
        self.size_2d = len(tar_files)
        self.size_3d = self.size_2d // self.frame
        print("Number of frames to load:", self.size_2d)
        print("Number of videos to load:", self.size_3d)
        
        inp_files_3d = list()
        tar_files_3d = list()
        for b in range(self.size_3d):
            inp_files_3d.append(inp_files[b * self.frame: (b + 1) * self.frame])
            tar_files_3d.append(tar_files[b * self.frame: (b + 1) * self.frame])
        
        self.inp_filenames = inp_files_3d
        self.tar_filenames = tar_files_3d
        
        
    def __len__(self):
        return self.size_3d

    def __getitem__(self, index):
        
        inp_array = np.array([np.load(inp_img_path) for inp_img_path in self.inp_filenames[index]])
        tar_array = np.array([np.load(tar_img_path) for tar_img_path in self.tar_filenames[index]])
        
        inp_array = inp_array[:, :, (256 - 128) // 2: (256 + 128) // 2, (256 - 128) // 2: (256 + 128) // 2]
        tar_array = tar_array[:, :, (256 - 128) // 2: (256 + 128) // 2, (256 - 128) // 2: (256 + 128) // 2]

        inp_tensors = torch.stack([torch.tensor(inp_array[i], dtype=torch.float32) for i in range(len(inp_array))])
        tar_tensors = torch.stack([torch.tensor(tar_array[i], dtype=torch.float32) for i in range(len(tar_array))])
        
        # for i in range(1):
        #     plt.imshow(inp_tensors[i][0, :, :], cmap='gray')
        #     plt.savefig('/root/autodl-tmp/TrainingSet/MultiCoil-yes/inp_img_' + str(i) + '.png') 
        #     plt.imshow(tar_tensors[i][0, :, :], cmap='gray')
        #     plt.savefig('/root/autodl-tmp/TrainingSet/MultiCoil-yes/tar_img_' + str(i) + '.png')     
        # print(inp_tensors.shape)
        # print(tar_tensors.shape)
        
        filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.inp_filenames[index]]
    
        return inp_tensors, tar_tensors, filenames

    
class DataLoaderVal(Dataset):
    def __init__(self, input_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        self.frame = img_options['frame']
        num = img_options['train_size']
        
        inp_files = list()
        tar_files = list()
        acc_AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10']
        ful_AFtype = ['FulSample04', 'FulSample08', 'FulSample10']
        p_set = p_set_list_val(num)
        for a in range(len(acc_AFtype)):
        # for a in range(1):
            inp_dirs = sorted(os.listdir(os.path.join(input_dir, acc_AFtype[a])))
            inp_files.extend([os.path.join(input_dir, acc_AFtype[a], x) for x in inp_dirs if is_require_npy(x, p_set)])
            tar_dirs = sorted(os.listdir(os.path.join(input_dir, ful_AFtype[a])))
            tar_files.extend([os.path.join(input_dir, ful_AFtype[a], x) for x in tar_dirs if is_require_npy(x, p_set)])
        
        self.size_2d = len(tar_files)
        self.size_3d = self.size_2d // self.frame
        # print("Number of frames to load:", self.size_2d)
        # print("Number of videos to load:", self.size_3d)
        
        inp_files_3d = list()
        tar_files_3d = list()
        for b in range(self.size_3d):
            inp_files_3d.append(inp_files[b * self.frame: (b + 1) * self.frame])
            tar_files_3d.append(tar_files[b * self.frame: (b + 1) * self.frame])
        
        self.inp_filenames = inp_files_3d
        self.tar_filenames = tar_files_3d
        
        
    def __len__(self):
        return self.size_3d

    def __getitem__(self, index):
        
        inp_array = np.array([np.load(inp_img_path) for inp_img_path in self.inp_filenames[index]])
        tar_array = np.array([np.load(tar_img_path) for tar_img_path in self.tar_filenames[index]])
        
        # print(inp_array.shape)
        inp_array = inp_array[:, :, (256 - 128) // 2: (256 + 128) // 2, (256 - 128) // 2: (256 + 128) // 2]
        tar_array = tar_array[:, :, (256 - 128) // 2: (256 + 128) // 2, (256 - 128) // 2: (256 + 128) // 2]

        inp_tensors = torch.stack([torch.tensor(inp_array[i], dtype=torch.float32) for i in range(len(inp_array))])
        tar_tensors = torch.stack([torch.tensor(tar_array[i], dtype=torch.float32) for i in range(len(tar_array))])
        
        # for i in range(1):
        #     plt.imshow(inp_tensors[i][0, :, :], cmap='gray')
        #     plt.savefig('/root/Shift-Net/inp_img_' + str(i) + '.png') 
        #     plt.imshow(tar_tensors[i][0, :, :], cmap='gray')
        #     plt.savefig('/root/Shift-Net/tar_img_' + str(i) + '.png')     
        # print(inp_tensors.shape)
        # print(tar_tensors.shape)
        
        filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.inp_filenames[index]]
    
        return inp_tensors, tar_tensors, filenames


if __name__ == "__main__":
    
    dir = '/root/autodl-tmp/TrainingSet/MultiCoil-yes'
    DataLoaderVal(dir, {'train_size': 5, 'frame': 12 * 1})[99]
#     datatrain = DataLoaderTrain(dir, {'train_size': 92, 'frame': 12 * 1})
    
#     from torch.utils.data import DataLoader
#     data = DataLoader(dataset=datatrain, batch_size=1, shuffle=True)
#     for i, da in enumerate(data):
#         inputs, targets, filenames = da
#         print(filenames)
#         if i == 2:
#             exit()
    
#     import glob
#     input_dir = '/root/autodl-tmp/TrainingSet/MultiCoil-yes'
#     acc_AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10']
#     ful_AFtype = ['FulSample04', 'FulSample08', 'FulSample10']
#     paths_acc = []
#     paths_ful = []
#     for a in range(1):
#         paths_acc.extend(sorted(glob.glob(os.path.join(input_dir, acc_AFtype[a], '*'))))
#         paths_ful.extend(sorted(glob.glob(os.path.join(input_dir, ful_AFtype[a], '*'))))
    
#     for i in range(12):
#         i = 12 * 1000 + i
#         inp_img = np.load(paths_acc[i])
#         tar_img = np.load(paths_ful[i])
#         plt.imshow(inp_img[0, :, :], cmap='gray')
#         plt.savefig('/root/autodl-tmp/TrainingSet/MultiCoil-yes/inp_img_' + str(i) + '.png')
#         plt.imshow(tar_img[0, :, :], cmap='gray')
#         plt.savefig('/root/autodl-tmp/TrainingSet/MultiCoil-yes/tar_img_' + str(i) + '.png')
    