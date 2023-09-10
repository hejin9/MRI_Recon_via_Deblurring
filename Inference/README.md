## Usage

The official pytorch implementation of the paper: **Accelerating Cardiac MRI via Deblurring without Sensitivity Estimation**


>其中my_checkpoint.pth文件太大，我放到了[百度网盘](https://pan.baidu.com/s/1l70_Q_Zq7xZhR00G6Qy_dg?pwd=ttwd) 提取码：ttwd 

首先将文件放到装好docker的linux服务器下，成功测试过的环境是ubuntu。

输入建立镜像的命令：

`docker build -t  docker.synapse.org/syn51751904/cine:v2 .`

其中 `docker.synapse.org/<Your project ID>/<taskname>:<Tag>` 可以另外取名字，`.` 代表在当前路径。

最后运行：

`docker run -it --rm --gpus device=0 -v "/my/input_data/path":"/input" -v "/my/output_data/path":"/output" docker.synapse.org/syn51751904/cine:v2`

一些docker镜像相关的命令：
```
docker image ls
docker ps -a
docker rm c7979af169f2
删除所有的container：docker rm $(docker ps -a -q)
删除所有的images：docker rmi $(docker images -q)
```