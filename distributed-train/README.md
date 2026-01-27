### 说明
ds-train.yaml/sh为多机lora微调deepseek-v3模型的配置文件、脚本。

40b-distributed-train.yaml/sh为集成muon优化器+deepspeed框架并用于多机分布式全量训练40B模型的配置文件、脚本，其中使用的镜像即为使用muon+deepspeed目录中代码构建好的镜像。