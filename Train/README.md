## Usage

The official pytorch implementation of the paper: **Accelerating Cardiac MRI via Deblurring without Sensitivity Estimation**

去模糊的方法、环境和代码参考：[DeepRFT](https://github.com/INVOKERer/DeepRFT)

### 改动的地方

数据预处理：
 - img_pre.py : 将原始*.mat文件转成*.np格式的二维图像域文件。

需要替换的文件有：
 - dataset_RGB.py : 加载的数据变化
 - DeepRFT_MIMO.py : 通道数从3变到了1
 - train.py : 一些参数改变