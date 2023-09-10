## MRI_Recon_via_Deblurring

Code for MICCAI2022 challenge CMRxRecon(Task1: Accelerated Cine reconstruction)

Challenge website: 
 - https://stacom.github.io/stacom2023/
 - https://cmrxrecon.github.io/

The official pytorch implementation of the paper: **Accelerating Cardiac MRI via Deblurring without Sensitivity Estimation**

**Abstract**：
>Reducing acquisition time in Cardiac Magnetic Resonance Imaging (MRI) brings several benefits, such as improved patient comfort, reduced motion artifacts and increased doctors' work efficiency, but it may lead to image blurring during the reconstruction process. 
In this paper, we propose a new method for restoring blurry cardiac MRI images caused by under-sampling, treating it as an image deblurring problem to achieve clear reconstruction, and ensuring consistency with training by using a simple modified input during inference.
A U-Net network architecture which initially designed for natural image deblurring, has been adapted to effectively discern the differences between blurred and clear MRI images, eliminating the need for sensitivity estimation. Moreover, to address the inconsistency between training on local patches and testing on the entire image, we propose a partial overlap cropping approach during inference time, effectively resolving this discrepancy. 
We evaluated our method using the cardiac MRI dataset from the CMRxRecon challenge, revealing its potential to reduce acquisition time while preserving high image quality in cardiac MRI, even under highly under-sampled conditions. Importantly, this achievement was attained in a coil-agnostic manner, enabling us to achieve favorable results on both multi-coil and single-coil data.
Our code is available at https://github.com/Hejin9/MRI_Recon_via_Deblurring.

![](./Problem%20Formulation.jpg)

## Usage

Inference 文件夹是最终提交给比赛的docker文件

train文件夹是训练时的文件