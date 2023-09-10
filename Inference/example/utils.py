from collections import OrderedDict
import numpy as np
import os
import torch



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_checkpoint_compress_doconv(model, weights):
    checkpoint = torch.load(weights)
    # print(checkpoint)
    # state_dict = OrderedDict()
    # try:
    #     model.load_state_dict(checkpoint["state_dict"])
    #     state_dict = checkpoint["state_dict"]
    # except:
    old_state_dict = checkpoint["state_dict"]
    state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        # print(k)
        name = k
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        state_dict[name] = v
    # state_dict = checkpoint["state_dict"]
    do_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[-1] == 'W' and k[:-1] + 'D' in state_dict:
            k_D = k[:-1] + 'D'
            k_D_diag = k_D + '_diag'
            W = v
            D = state_dict[k_D]
            D_diag = state_dict[k_D_diag]
            D = D + D_diag
            # W = torch.reshape(W, (out_channels, in_channels, D_mul))
            out_channels, in_channels, MN = W.shape
            M = int(np.sqrt(MN))
            DoW_shape = (out_channels, in_channels, M, M)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            do_state_dict[k] = DoW
        elif k[-1] == 'D' or k[-6:] == 'D_diag':
            continue
        elif k[-1] == 'W':
            out_channels, in_channels, MN = v.shape
            M = int(np.sqrt(MN))
            W_shape = (out_channels, in_channels, M, M)
            do_state_dict[k] = torch.reshape(v, W_shape)
        else:
            do_state_dict[k] = v
    model.load_state_dict(do_state_dict)