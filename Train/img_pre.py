
# 将mat的复数域格式文件，转成图像域的numpy格式文件


import os
import numpy as np
import matplotlib.pyplot as plt
import h5py



def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['sax.mat', 'lax.mat'])

def get_filepaths(train_dir, imgtype):
    assert os.path.exists(os.path.join(train_dir, imgtype))

    inp_paths = sorted(os.walk(os.path.join(train_dir, imgtype)))

    inp_files = []

    for path, dir_lst, file_lst in inp_paths:
        for file_name in file_lst:
            if is_mat_file(file_name):
                inp_files.append(os.path.join(path, file_name))

    return inp_files


def read_data(inp_path, out_path, is_singlecoil, is_Val):

    # kspace: complex images with the dimensions (sx, sy, sc, sz, frame)
    # when is_singlecoil = 1, kspace.shape = (sx, sy, sz, frame)
    # sx: matrix size in x-axis
    # sy: matrix size in y-axis
    # sc: coil array number
    # sz: slice number (short axis view); slice group (long axis view)
    # t:  time frame

    filenames = [None for _ in range(4)]
    filenames[0] = os.path.split(os.path.split(os.path.split(inp_path)[0])[0])[-1]
    filenames[1] = os.path.split(inp_path)[-1]
    filenames[1] = filenames[1].split('.')[0]
    filenames[2] = os.path.split(os.path.split(inp_path)[0])[-1]

    # AccFactor04-P001-cine_sax.mat

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


    if is_singlecoil:

        dic = {
            'AccFactor04': 'kspace_single_sub04',
            'AccFactor08': 'kspace_single_sub08',
            'AccFactor10': 'kspace_single_sub10',
            'FullSample': 'kspace_single_full'
        }

        rawdata = h5py.File(inp_path, 'r')
        kspace_raw = np.array(rawdata[dic[filenames[0]]]).transpose((3, 2, 1, 0))
        kspace = kspace_raw['real'] + kspace_raw['imag'] * 1j
        kspace = np.array(kspace, dtype=np.complex64)
        sx, sy, sz, t = kspace.shape

        print(kspace.shape)

        img_full = np.zeros((sx, sy, sz, t), dtype=np.float64)
        for i in range(sz):
            for j in range(t):
                img_full[:, :, i, j] = Ifft2c(kspace[:, :, i, j])

                filename = filenames[0] + '-' + filenames[2] + '-' + filenames[1] + '-' \
                                       + str(i + 1).zfill(2) + '-' + str(j + 1).zfill(2) + '.npy'
                filepath = os.path.join(out_path, filename)
                print(filepath)
                np.save(filepath, img_full[:, :, i, j])

    else:

        dic1 = {
            'AccFactor04': 'kspace_sub04',
            'AccFactor08': 'kspace_sub08',
            'AccFactor10': 'kspace_sub10',
            'FullSample': 'kspace_full'
        }

        rawdata = h5py.File(inp_path, 'r')
        kspace_raw = np.array(rawdata[dic1[filenames[0]]]).transpose((4, 3, 2, 1, 0))
        kspace = kspace_raw['real'] + kspace_raw['imag'] * 1j
        kspace = np.array(kspace, dtype=np.complex64)
        sx, sy, sc, sz, t = kspace.shape

        print(kspace.shape)

        img_full_sos = np.zeros((sx, sy, sc, sz, t), dtype=np.complex64)
        img_full = np.zeros((sx, sy, sz, t), dtype=np.float64)
        for i in range(sz):
            for j in range(t):
                img_full_sos[:, :, :, i, j] = Ifft2c(kspace[:, :, :, i, j])
                img_full[:, :, i, j] = sos(img_full_sos[:, :, :, i, j])

                filename = filenames[0] + '-' + filenames[2] + '-' + filenames[1] + '-' \
                           + str(i + 1).zfill(2) + '-' + str(j + 1).zfill(2) + '.npy'
                filepath = os.path.join(out_path, filename)
                # AccFactor04-P001-cine_lax-03-12.npy

                np.save(filepath, img_full[:, :, i, j])
                print(filepath)   # 网络输出图片时可以根据名字存到对应的文件夹和mat文件中



if __name__ == "__main__":


    TrainorVal = ['TrainingSet', 'ValidationSet']
    t = 1
    is_Val = t

    for c in range(2):      # 0, 1
        for a in range(3):  # 0, 1, 2, (3)
            inpdir = 'D:/data/CMRxRecon/MICCAIChallenge2023/ChallengeData'
            outdir = 'D:/data/CMRxRecon'

            AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10',
                      'FullSample']  # only 'TrainingSet' has 'FullSample'
            imgtype = AFtype[a]
            coilInfo = ['MultiCoil', 'SingleCoil']
            is_singlecoil = c

            inp_dir = os.path.join(inpdir, coilInfo[c], 'Cine', TrainorVal[t])
            # 'D:/data/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil/Cine/TrainingSet'
            out_path = os.path.join(outdir, TrainorVal[t], coilInfo[c], imgtype)
            # 'D:/data/CMRxRecon/TrainingSet/MultiCoil/AccFactor04'

            inp_filepaths = get_filepaths(inp_dir, imgtype)
            # print(inp_filenames)

            # __getitem__
            # index = 0

            for index in range(len(inp_filepaths)):

                inp_path = inp_filepaths[index]

                print(inp_path)
                # print(out_path)

                inp_img = read_data(inp_path, out_path, is_singlecoil, is_Val)
