
# 说明

本文件夹中的代码在llamafactory的基础上，集成了muon优化器与deepspeed框架（原始的llamafactory不同时支持这两个功能），可以用这两个功能在**单机**上全量训练40B模型。在llamafactory的原始框架上进行修改的具体过程位于**单机代码修改过程.txt**与**多机代码修改过程.txt**中，构建镜像的过程在**镜像构建.txt**中。

**注意：在包头集群上，脚本中的文件的路径需要是：/mnt/workspace/wanghao277/... 而非/mnt/jpfs-5p/wanghao277/...**

## 启动单机训练

1.进入muon+deepspeed目录

1.修改数据集的配置文件，路径为data/dataset_info.json

2.修改配置文件40b-sft-full.yaml中的模型、数据路径以及其他参数

3.运行下面命令，进行**单机全量训练**：

```bash

llamafactory-clitrain40b-sft-full.yaml

```

## 备注

同样功能的多机训练的代码在distributed-train文件夹中

在多机训练模型时，参考**多机代码修改过程.txt**底部，仿照其方式，**修改你使用的模型文件目录中的代码！**，详见其中的README
