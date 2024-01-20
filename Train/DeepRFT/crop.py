from decimal import Decimal
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from scipy.io import savemat, loadmat


MultiCoil = True
SingleCoil = False

input_dir_org = './Outputs'
output_dir_org = './Outputs_Crop/MultiCoil/Cine/ValidationSet'



AFtype = ['AccFactor04', 'AccFactor08', 'AccFactor10']
dic1 = {
            'AccFactor04': 'kspace_sub04',
            'AccFactor08': 'kspace_sub08',
            'AccFactor10': 'kspace_sub10',
            'FullSample': 'kspace_full'
        }
dic2 = {
            'AccFactor04': 'kspace_single_sub04',
            'AccFactor08': 'kspace_single_sub08',
            'AccFactor10': 'kspace_single_sub10',
            'FullSample': 'kspace_single_full'
        }

if MultiCoil == True:
    input_dir = os.path.join(input_dir_org, 'MultiCoil/Cine/ValidationSet')
    dic = dic1
if SingleCoil == True:
    input_dir = os.path.join(input_dir_org, 'SingleCoil/Cine/ValidationSet')
    dic = dic2

output_dir = './Outputs_Crop/MultiCoil/Cine/ValidationSet'


for f in range(3):  # 选AFtype
    for p in tqdm(range(60), desc=f"Accfactor {f + 1} th"):
        # for p in range(1):  # 选 1 - 60的数
        patient = 'P' + str(p + 1).zfill(3)
        inp_dir = os.path.join(input_dir, AFtype[f], patient)
        tar_dir = os.path.join(output_dir, AFtype[f], patient)
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)

        file_list = glob.glob(os.path.join(inp_dir, '*ax.mat'))

        for file in file_list:

            # print('\noutput_dir:', file)
            filename = os.path.split(file)[-1]
            tar_dir_filename = os.path.join(tar_dir, filename)

            rawdata =loadmat(file)
            inp_img = np.array(rawdata[dic[AFtype[f]]], dtype=np.float32)
            sx, sy, sz, t = inp_img.shape
            # tar_img = np.empty((None, None, 2, 3), dtype=np.float32)

            def compute_index(sx, sy, sz):
                m = list()
                s = list()
                m.append(sx)
                m.append(sy)
                s.append(Decimal(sx / 3).quantize(Decimal("1."), rounding="ROUND_HALF_UP"))
                s.append(Decimal(sy / 2).quantize(Decimal("1."), rounding="ROUND_HALF_UP"))
                a = int(np.floor(m[0] / 2) + np.ceil(-s[0] / 2))
                b = int(np.floor(m[0] / 2) + np.ceil(s[0] / 2))
                c = int(np.floor(m[1] / 2) + np.ceil(-s[1] / 2))
                d = int(np.floor(m[1] / 2) + np.ceil(s[1] / 2))
                g = (sz + 1) // 2 - 2 if sz != 2 else 0
                h = (sz + 1) // 2 if sz != 2 else 2
                return a, b, c, d, g, h

            i = 2
            j = 3

            a, b, c, d, g, h = compute_index(sx, sy, sz)
            tar_img = inp_img[a:b, c:d, g:h, :j]
            # print(tar_img.shape)
            # print(tar_dir_filename)
            # print(dic[AFtype[f]])
            savemat(tar_dir_filename, {'img4ranking': tar_img})

            # for i in range(0, 2):
            #     for j in range(0, 3):
            #         plt.imshow(inp_img[:, :, g + i, j], vmin=0, vmax=0.001, cmap='gray')
            #         plt.show()
            #         plt.imshow(tar_img[:, :, i, j], vmin=0, vmax=0.001, cmap='gray')
            #         plt.show()

    pass
