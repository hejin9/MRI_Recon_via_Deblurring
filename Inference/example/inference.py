import argparse
import glob
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import savemat
import torch
from tqdm import tqdm

from Model.DeepRFT_MIMO import DeepRFTPLUS
from utils import load_checkpoint_compress_doconv

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
    parser.add_argument('--input', type=str, default='./input', help='input directory')
    # /input, D:/data/CMRxRecon/MICCAIChallenge2023/ChallengeData
    parser.add_argument('--output', type=str, default='./output', help='output directory')
    # /output, ./output
    parser.add_argument('--weights', type=str, default='/app/my_checkpoint.pth')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    input = args.input
    output = args.output

    input_dir = os.path.join(input, 'MultiCoil/Cine/TestSet')
    output_dir = os.path.join(output, 'MultiCoil/Cine/TestSet')

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    # model_restoration = Net()
    model_restoration = DeepRFTPLUS(num_res=20, inference=True)

    load_checkpoint_compress_doconv(model_restoration, args.weights)
    print('===> Testing using weights: ', args.weights)

    model_restoration.cuda()
    model_restoration.eval()

    AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10']
    dic1 = {
        'AccFactor04': 'kspace_sub04',
        'AccFactor08': 'kspace_sub08',
        'AccFactor10': 'kspace_sub10',
        'FullSample': 'kspace_full'
    }


    for a in range(3):
        for p in tqdm(range(120), desc=f"{AFtype[a]} "):

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

                    inp_img = torch.tensor(img_full[:, :, i:i+1, :], dtype=torch.float32).permute(3, 2, 0, 1)

                    with torch.no_grad():
                        torch.cuda.empty_cache()

                        sx_n = (64 - sx % 64) % 64
                        sy_n = (64 - sy % 64) % 64

                        shadow = 8   # 重叠的部分
                        mod = 128 - shadow
                        sx_n = (mod - (sx - shadow) % mod) % mod
                        sy_n = (mod - (sy - shadow) % mod) % mod
                        input_re = torch.nn.functional.pad(inp_img, (0, sy_n, 0, sx_n), mode='reflect')

                        n = input_re.shape[2]
                        m = input_re.shape[3]
                        k = shadow

                        num = input_re.shape[0]
                        nx = (n - k) // (128 - k)
                        ny = (m - k) // (128 - k)

                        tmp_input = torch.zeros(input_re.shape[0] * nx * ny, 1, 128, 128)
                        for ii in range(input_re.shape[0]):
                            for x in range(nx):
                                for y in range(ny):
                                    trueX = x * (128 - k)
                                    trueY = y * (128 - k)
                                    tmp_input[ii * nx * ny + x * ny + y, 0, :, :] = input_re[ii, 0, trueX : trueX + 128, trueY : trueY + 128]
                        input_re = tmp_input

                        restored = model_restoration(input_re.cuda())

                        tmp_restored = torch.zeros(num, 1, n, m)
                        for ii in range(tmp_restored.shape[0]):
                            for x in range(nx):
                                for y in range(ny):
                                    trueX = x * (128 - k)
                                    trueY = y * (128 - k)
                                    tmp_restored[ii, 0, trueX : trueX + 128, trueY : trueY + 128] = restored[ii * nx * ny + x * ny + y, 0, :, :]

                        restored = tmp_restored
                        restored = restored[:, :, :sx, :sy].permute(2, 3, 1, 0)
                        restored = restored.cpu().detach().numpy()

                    tar_img[:, :, i, :] = restored[:, :, 0, :]

                savemat(tar_dir_filename, {'img4ranking': tar_img})

        pass