### 注意事项
本文件夹中的代码在llamafactory的基础上集成了muon优化器与deepspeed框架，可以用这两个功能全量训练40B模型。在llamafactory的原始框架上进行修改的过程位于“代码修改过程.txt”中。
**注意** 
配置文件为当前目录下的40b-sft-full.yaml，具体使用方式为：
1.修改数据集的配置文件，路径为data/dataset_info.json
2.修改配置文件40b-sft-full.yaml中的模型、数据路径以及其他参数
3.运行下面命令，进行训练：
```bash
llamafactory-cli train 40b-sft-full.yaml
```
**注意**