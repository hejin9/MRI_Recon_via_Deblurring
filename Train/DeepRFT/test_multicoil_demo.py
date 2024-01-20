

import argparse
import glob
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.io import savemat
import torch
import torch.nn as nn
from tqdm import tqdm

from DeepRFT_MIMO import DeepRFT
from get_parameter_number import get_parameter_number
from utils import load_checkpoint_compress_doconv

def window_partition(input, window_size):
    _, _, h, w = input.shape
    a = math.ceil(h / window_size)
    b = math.ceil(w / window_size)
    H, W = a * window_size, b * window_size

    if H > h or W > w:
        input = torch.nn.functional.pad(input, [0, W - w, 0, H - h], mode='reflect')

    tmp = np.zeros((a * b, 1, window_size, window_size))
    tmp = torch.tensor(tmp, dtype=torch.float32)
    index = 0
    for i in range(a):
        for j in range(b):
            tmp[index, 0, :, :] = input[0, 0, i*window_size: (i+1)*window_size, j*window_size: (j+1)*window_size]
            index = index + 1

    tmp = tmp.contiguous().view(index, 1, window_size, window_size)

    return tmp

def window_reverse(output, window_size, h, w):
    a = math.ceil(h / window_size)
    b = math.ceil(w / window_size)
    H, W = window_size * math.ceil(h / window_size), window_size * math.ceil(w / window_size)

    tmp = np.zeros((1, 1, H, W))
    tmp = torch.tensor(tmp, dtype=torch.float32)
    index = 0
    for i in range(a):
        for j in range(b):
            tmp[0, 0, i * window_size: (i + 1) * window_size, j * window_size: (j + 1) * window_size] \
                = output[index, 0, :, :]
            index = index + 1

    tmp = tmp[:, :, :h, :w].contiguous().view(1, 1, h, w)

    return tmp

def Ifft2c(x):
    if len(x.shape) == 2:
        S = x.shape
        fctr = S[0] * S[1]
        res = np.zeros_like(x)
        res[:, :] = np.sqrt(fctr) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x[:, :])))
        res = np.abs(res)

    else:
        S = x.shape
        fctr = S[0] * S[1]
        x = np.reshape(x, (S[0], S[1], np.prod(S[2:])))
        res = np.zeros_like(x)

        for n in range(x.shape[2]):
            res[:, :, n] = np.sqrt(fctr) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x[:, :, n])))

        res = np.reshape(res, S)
    return res

def sos(x):
    # sos: compute sum-of-squares
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parallel Imaging')
    parser.add_argument('--input_dir', type=str, default='D:/data/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil/Cine/ValidationSet')
    parser.add_argument('--output_dir', default='./Outputs/MultiCoil/Cine/ValidationSet', type=str)
    parser.add_argument('--win_size', type=int, default=256)
    parser.add_argument('--weights', type=str,
                        default='../checkpoints/mul-origin-0702/model-10_ssim_best.pth')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    input_dir = args.input_dir
    output_dir = args.output_dir
    win_size = args.win_size

    # model_restoration = Net()
    model_restoration = DeepRFT(num_res=8, inference=True)

    # # utils.load_checkpoint(model_restoration, args.weights)
    load_checkpoint_compress_doconv(model_restoration, args.weights)
    print('===> Testing using weights: ', args.weights)

    model_restoration.cuda()
    # model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10']
    dic1 = {
        'AccFactor04': 'kspace_sub04',
        'AccFactor08': 'kspace_sub08',
        'AccFactor10': 'kspace_sub10',
        'FullSample': 'kspace_full'
    }
    for a in range(3):      # 选AFtype
        for p in tqdm(range(60), desc=f"Accfactor {a + 1} th"):
        # for p in range(1):  # 选 1 - 60的数
            patient = 'P' + str(p + 1).zfill(3)
            inp_dir = os.path.join(input_dir, AFtype[a], patient)
            tar_dir = os.path.join(output_dir, AFtype[a], patient)
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)

            file_list = glob.glob(os.path.join(inp_dir, '*ax.mat'))

            for file in file_list:

                # print('\noutput_dir:', file)
                filename = os.path.split(file)[-1]
                tar_dir_filename = os.path.join(tar_dir, filename)

                rawdata = h5py.File(file, 'r')
                k_raw = np.array(rawdata[dic1[AFtype[a]]]).transpose((4, 3, 2, 1, 0))
                k_space = k_raw['real'] + k_raw['imag'] * 1j
                k_space = np.array(k_space, dtype=np.complex64)
                sx, sy, sc, sz, t = k_space.shape

                img_full_sos = np.zeros((sx, sy, sc, sz, t), dtype=np.complex64)
                img_full = np.zeros((sx, sy, sz, t), dtype=np.float32)
                tar_img = np.zeros((sx, sy, sz, t), dtype=np.float32)

                for i in range(sz):
                    for j in range(t):
                        img_full_sos[:, :, :, i, j] = Ifft2c(k_space[:, :, :, i, j])
                        img_full[:, :, i, j] = sos(img_full_sos[:, :, :, i, j])

                        tmp = img_full[:, :, i, j]
                        inp_img = np.zeros((1, 1, sx, sy))
                        inp_img[0, 0, :, :] = tmp
                        # plt.imshow(inp_img, vmin=0, vmax=0.001, cmap='gray')
                        # plt.show()
                        with torch.no_grad():
                            torch.cuda.ipc_collect()
                            torch.cuda.empty_cache()

                            # inp_img = torch.tensor(inp_img, dtype=torch.float32)
                            # input_re = window_partition(inp_img, win_size)
                            #
                            # restored = model_restoration(input_re.cuda())
                            #
                            # restored = window_reverse(restored, win_size, sx, sy)
                            # restored = restored.cpu().detach().numpy()

                            sx_n = (32 - sx % 32) % 32
                            sy_n = (32 - sy % 32) % 32
                            inp_img = torch.tensor(inp_img, dtype=torch.float32)
                            input_re = torch.nn.functional.pad(inp_img, (0, sy_n, 0, sx_n), mode='reflect')
                            restored = model_restoration(input_re.cuda())
                            restored = restored[:, :, :sx, :sy]

                            restored = restored.cpu().detach().numpy()

                            plt.imshow(restored[0, 0, :, :], vmin=0, vmax=0.001, cmap='gray')
                            plt.show()
                            exit()

                        tar_img[:, :, i, j] = restored[0, 0, :, :]

                savemat(tar_dir_filename, {dic1[AFtype[a]]: tar_img})

        pass