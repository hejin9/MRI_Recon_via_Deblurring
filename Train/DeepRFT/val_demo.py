

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DeepRFT_MIMO import DeepRFT_Small, DeepRFT, DeepRFTPLUS
from mydataset_val import DataLoaderVal
from get_parameter_number import get_parameter_number
# from layers import window_partition, window_reverse
from utils import load_checkpoint_compress_doconv, mkdir





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parallel Imaging')
    parser.add_argument('--input_dir', default='/public/home/hejin/TrainingSet/MultiCoil-yes', type=str)
    parser.add_argument('--weights', default='./model_latest_0626.pth', type=str)
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_res', default=8, type=int, help='num of resblocks, [Small, Med, PLus]=[4, 8, 20]')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    input_dir = args.input_dir


    # model_restoration = Net()
    model_restoration = DeepRFT(num_res=args.num_res, inference=True)

    # print number of model
    # get_parameter_number(model_restoration)

    # utils.load_checkpoint(model_restoration, args.weights)
    load_checkpoint_compress_doconv(model_restoration, args.weights)
    print('===> Testing using weights: ', args.weights)

    model_restoration.cuda()
    model_restoration.eval()

    # dataset
    test_dataset = DataLoaderVal(input_dir, img_options={'train_size': 5, 'frame': 12})
    # test_dataset = DataLoaderTest(input_dir, img_options={'train_size': 100})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # data = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # for i, da in enumerate(data):
    #     inputs, targets, filenames = da
    #     if i > 61:
    #         print(filenames[0])
    #     if i == 68:
    #         exit()

    with torch.no_grad():

        ssim_val = []
        psnr_val = []
        nmse_val = []

        for ii, data_test in enumerate(tqdm(test_loader), 0):

            torch.cuda.empty_cache()


            target = data_test[1][0]
            input_ = data_test[0][0]
            filenames = data_test[2]
            
            # target = torch.cat((target[0].unsqueeze(0), target))
            # input_ = torch.cat((input_[0].unsqueeze(0), input_))
#             print('*********')
#             print(data_test[2])
            
            
#             print(filenames)
#             print(len(filenames))
#             filenames = [filenames[0], filenames]
#             print('file:', len(filenames))
#             print(filenames)
#             exit()

            # input_re = window_partition(input_, win_size)

            # restored = model_restoration(input_re.cuda())

            # restored = window_reverse(restored, win_size, Hx, Wx)
            
            # print('**************')
            # print(target.shape)
            # print(input_.shape)
            restored = model_restoration(input_.cuda())
            restored = restored.cpu().detach().numpy()

            # plt.imshow(input_[0, 0, :, :], vmin=0, vmax=0.001, cmap='gray')
            # plt.show()
            # plt.imshow(target[0, 0, :, :], vmin=0, vmax=0.001, cmap='gray')
            # plt.show()
            # plt.imshow(restored[0, 0, :, :], vmin=0, vmax=0.001, cmap='gray')
            # plt.show()
            
            # print(restored.shape)
            # print(restored[1:].shape)
            # exit()
            
            for res, tar, name in zip(restored, target, filenames):
                
                res = res[0, :, :]
                tar = tar[0, :, :].numpy()
                tar = tar / tar.max()
                res = res / res.max() if res.max() != 0 else res

                maxval = tar.max()
                psnr_val.append(psnr(tar, res, data_range=maxval).item())
                ssim_val.append(ssim(tar, res, data_range=maxval).item())
                nmse_val.append(np.array(np.linalg.norm(tar - res) ** 2 / np.linalg.norm(tar) ** 2).item())

                # print(psnr_val[-1], ssim_val[-1], nmse_val[-1])
                if ssim_val[-1] < 0.85:
                    print(ssim_val[-1], name)


        psnr_val = np.mean(psnr_val)
        ssim_val = np.mean(ssim_val)
        nmse_val = np.mean(nmse_val)

        print('total: ', psnr_val, ssim_val, nmse_val)

