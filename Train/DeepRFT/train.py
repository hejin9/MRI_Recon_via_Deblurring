import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from DeepRFT_MIMO import DeepRFT as myNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter
import argparse

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


######### Set Seeds ###########
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed_all(2023)

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deblurring')

parser.add_argument('--train_dir', default='/public/home/hejin/TrainingSet/MultiCoil', type=str, help='Directory of train images')
parser.add_argument('--log_save_dir', default='/public/home/hejin/my-log/deep-0823', type=str, help='Path to save weights')
parser.add_argument('--model_save_dir', default='/public/home/hejin/my-cp/deep-0823', type=str, help='Path to save weights')
parser.add_argument('--model_pre_dir', default='/public/home/hejin/my-cp/deep-0724/model_ssim-0.8927682000350533_iter-990670.pth', type=str)
parser.add_argument('--patch_size', default=128, type=int, help='patch size')
parser.add_argument('--num_epochs', default=80, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--train_size', default=115, type=int, help='train_size')
parser.add_argument('--val_size', default=5, type=int, help='val_epochs')
args = parser.parse_args()

patch_size = args.patch_size

log_dir = os.path.join(args.log_save_dir)
utils.mkdir(log_dir)
model_dir = os.path.join(args.model_save_dir)
utils.mkdir(model_dir)

train_dir = args.train_dir

num_epochs = args.num_epochs
batch_size = args.batch_size
train_size = args.train_size
val_size = args.val_size

start_lr = 1e-4
end_lr = 1e-6

######### Model ###########
model_restoration = myNet()

# print number of model
get_parameter_number(model_restoration)

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = False
Pretrain = True
model_pre_dir = args.model_pre_dir
######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration, model_pre_dir)

    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with: " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()

######### DataLoaders ###########
print('===> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': patch_size, 'train_size': train_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dir = '/public/home/hejin/TrainingSet/MultiCoil'
val_dataset = get_validation_data(val_dir, {'patch_size': 128, 'train_size': 5})
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs))

best_ssim = [0.87, 0.87, 0.87]
best_iter = [0, 0, 0]
best_psnr = 0
writer = SummaryWriter(log_dir)
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda()
        input_  = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(
            restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(restored[2], target[2])
        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        iter += 1
        writer.add_scalar('loss/iter_loss', loss, iter)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
    #### Evaluation ####
    if epoch % 1 == 0:
        model_restoration.eval()
        ssim_val = []
        psnr_val = []
        nmse_val = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)

            for res,tar in zip(restored[0], target):
                res = res[0, :, :].cpu().detach().numpy()
                tar = tar[0, :, :].cpu().detach().numpy()
                tar = tar / tar.max()
                res = res / res.max()  if res.max() != 0 else res

                maxval = tar.max()
                psnr_val.append(psnr(tar, res, data_range=maxval).item())
                ssim_val.append(ssim(tar, res, data_range=maxval).item())
                nmse_val.append(np.array(np.linalg.norm(tar - res) ** 2 / np.linalg.norm(tar) ** 2).item())
                # print("######################")
                # print(psnr_val[-1])
                # print(ssim_val[-1])
                # print(nmse_val[-1])

        psnr_val = np.mean(psnr_val)
        ssim_val = np.mean(ssim_val)
        nmse_val = np.mean(nmse_val)

        writer.add_scalar('val/psnr', psnr_val, epoch)
        writer.add_scalar('val/ssim', ssim_val, epoch)
        writer.add_scalar('val/nmse', nmse_val, epoch)
        
        
        tmpl = []
        for i in range(len(best_ssim)):
            tmpl.append([best_ssim[i], best_iter[i]])
        tmpl.append([ssim_val, iter])
        tmpl.sort(key = lambda x: x[0], reverse = True)   
        remove_item = tmpl[-1]
        tmpl.pop()
        print(tmpl)

        for i in range(len(tmpl)):
            best_ssim[i] = tmpl[i][0]
            best_iter[i] = tmpl[i][1]

            if tmpl[i][1] == iter:
                torch.save({'epoch': epoch, 
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir, f"model_ssim-{ssim_val}_iter-{iter}.pth"))

        # if ssim_val > best_ssim:
        #     best_ssim = ssim_val
        #     best_iter = iter
        #     torch.save({'iter': iter, 
        #                 'state_dict': model_restoration.state_dict(),
        #                 'optimizer' : optimizer.state_dict()
        #                 }, os.path.join(model_dir, f"model_ssim-{ssim_val}_iter-{iter}.pth"))

        if psnr_val > best_psnr:
            best_psnr = psnr_val

            
        print("epoch %d SSIM: %.4f PSNR: %.4f NMSE: %.4f --- best_iter %d Best_SSIM %.4f (best_psnr %.4f)" % (epoch, ssim_val, psnr_val, nmse_val, tmpl[0][1], tmpl[0][0], best_psnr))
        
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}_{ssim_val}.pth")) 
        

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 

writer.close()
