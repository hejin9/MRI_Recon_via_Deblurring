from mydataset import *
import math


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)



import matplotlib.pyplot as plt
def window_partition(input, window_size):
    siz, _, h, w = input.shape
    # print('input.shape:', input.shape)
    a = math.ceil(h / window_size)
    b = math.ceil(w / window_size)
    H, W = a * window_size, b * window_size

    if H > h or W > w:
        tmp2 = torch.nn.functional.pad(input, [0, W - w, 0, H - h], mode='reflect')
    
    input = input
    # print('input.shape:', input.shape)
    # plt.imshow(input[0, 0, :, :], cmap='gray')
    # plt.savefig('pad_img')
    
    tmp = np.zeros((a * b * siz, 1, window_size, window_size))
    tmp = torch.tensor(tmp, dtype=torch.float32)
    index = 0
    for k in range(siz):
        for i in range(a):
            for j in range(b):
                tmp[index, 0, :, :] = tmp2[k, 0, i*window_size: (i+1)*window_size, j*window_size: (j+1)*window_size]
                index = index + 1
    
    input = tmp.contiguous().view(index, 1, window_size, window_size)
    
    # plt.imshow(tmp[0, 0, :, :], cmap='gray')
    # plt.savefig('crop1_img')
    # plt.imshow(tmp[1, 0, :, :], cmap='gray')
    # plt.savefig('crop2_img')
    # plt.imshow(tmp[index - 2, 0, :, :], cmap='gray')
    # plt.savefig('crop3_img')
    # plt.imshow(tmp[index - 1, 0, :, :], cmap='gray')
    # plt.savefig('crop4_img')
    # exit()
    
    return input

def window_reverse(output, window_size, s, h, w):
    a = math.ceil(h / window_size)
    b = math.ceil(w / window_size)
    H, W = window_size * math.ceil(h / window_size), window_size * math.ceil(w / window_size)

    tmp = np.zeros((s, 1, H, W))
    tmp = torch.tensor(tmp, dtype=torch.float32)
    index = 0
    for k in range(s):
        for i in range(a):
            for j in range(b):
                tmp[k, 0, i * window_size: (i + 1) * window_size, j * window_size: (j + 1) * window_size] \
                    = output[index, 0, :, :]
                index = index + 1

    tmp = tmp[:, :, :h, :w].contiguous().view(s, 1, h, w)
    
    # plt.imshow(tmp[0, 0, :, :], cmap='gray')
    # plt.savefig('rest1_img')
    # plt.imshow(tmp[-1, 0, :, :], cmap='gray')
    # plt.savefig('rest2_img')
    # exit()

    return tmp