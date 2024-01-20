import matplotlib.pyplot as plt
import numpy as np
from DeepRFT_MIMO import DeepRFT
import torch
from utils import load_checkpoint_compress_doconv
model_restoration = DeepRFT(num_res=8, inference=True)
load_checkpoint_compress_doconv(model_restoration, '/public/home/hejin/my-cp/deep-0714/model_latest.pth')

model_restoration.cuda()
model_restoration.eval()

input_dir = '/public/home/hejin/TrainingSet/MultiCoil-yes/AccFactor04/AccFactor04-P096-cine_sax-08-01.npy'
targt_dir = '/public/home/hejin/TrainingSet/MultiCoil-yes/FulSample04/FulSample04-P096-cine_sax-08-01.npy'
input_ = np.load(input_dir)
input_ = torch.tensor(np.expand_dims(input_, axis=0), dtype=torch.float32)
tar_img = np.load(targt_dir)

restored = model_restoration(input_.cuda())
restored = restored.cpu().detach().numpy()

plt.imshow(restored[0, 0, :, :], cmap='gray')
plt.savefig('./inp_img1.png')
plt.imshow(tar_img[0, :, :], cmap='gray')
plt.savefig('./tar_img.png')

print('restored.min():', restored.min())
res_flat = restored[0, 0, :, :].flatten()
print(res_flat)
plt.hist(res_flat)
plt.savefig('./hist_1.png')

tmp = restored[0, 0, :, :] - tar_img[0, :, :]
print(restored[0, 0, 0, 0], tar_img[0, 0, 0], restored[0, 0, 0, 0] - tar_img[0, 0, 0])
# print(restored[0, 0, 0:10, 0:10])
# print(restored[0, 0:10, 0:10])
# print(tmp[0:10, 0:10])

# tmp = tmp - tmp.min()/(tmp.max() - tmp.min())
# plt.imshow(tmp, cmap='gray')
# plt.savefig('./tmp_img.png')
# print(tmp[0:10, 0:10])
load_checkpoint_compress_doconv(model_restoration, './model_latest_0626.pth')

model_restoration.cuda()
model_restoration.eval()

input_dir = '/public/home/hejin/TrainingSet/MultiCoil-yes/AccFactor04/AccFactor04-P096-cine_sax-08-01.npy'
targt_dir = '/public/home/hejin/TrainingSet/MultiCoil-yes/FulSample04/FulSample04-P096-cine_sax-08-01.npy'
input_ = np.load(input_dir)
input_ = torch.tensor(np.expand_dims(input_, axis=0), dtype=torch.float32)
tar_img = np.load(targt_dir)

restored = model_restoration(input_.cuda())
restored = restored.cpu().detach().numpy()
plt.imshow(restored[0, 0, :, :], cmap='gray')
plt.savefig('./inp_img2.png')

print('restored.min():', restored.min())
res_flat = restored[0, 0, :, :].flatten()
plt.hist(res_flat, bins=256, color='gray', alpha=0.8)
plt.savefig('./hist_2.png')


# 创建一个形状为 (128, 128) 的二维数组，数值范围为 0-0.008 之间
image = np.random.rand(128, 128)

# 将二维数组展开为一维数组
image_flat = image.flatten()

# 绘制直方图
plt.hist(image_flat, bins=256, color='gray', alpha=0.8)

# 设置坐标轴标签和标题
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Image')

# 保存直方图到本地
plt.savefig('./hist_3.png')