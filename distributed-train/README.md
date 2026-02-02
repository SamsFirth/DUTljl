
### 说明

ds-train.yaml/sh为基于megatron框架，多机lora微调deepseek-v3模型的配置文件、脚本。

40b-distributed-train.yaml/sh为基于llamafactory框架，集成muon优化器+deepspeed并用于**多机分布式**全量训练40B模型的配置文件、脚本，其中使用的镜像即为使用muon+deepspeed目录中代码构建好的镜像。

1.两个脚本的前半部分一致，包括配置环境变量、清理transformers_modules缓存、建立日志目录等环节，区别在于具体训练启动方式（二者所用框架不同）

2.注意ds-train.sh中的多机lora训练启动方式，使用ms-swift与megatron框架：

```bash

megatronsft\

  ...

--tensor_model_parallel_size8\

  --expert_model_parallel_size 16 \

--context_parallel_size4\

  --sequence_parallel true \

--pipeline_model_parallel_size1\

```

开启了张量并行、专家并行、上下文并行、序列并行，没有开启流水线并行，能够训练超大模型

3.注意40b-distributed-train.sh中的muon+deepspeed多机全量训练启动方式，使用llamafactory框架：

```bash

FORCE_TORCHRUN=1NNODES=${NUM_NODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \

  llamafactory-cli train \

--model_name_or_path...\

    --dataset_dir ... \

...

```

使用torchrun命令启动分布式训练，符合llamafactory官方的启动方式

使用的deepspeed配置为zero-2，使用zero-3会存在训练速度很慢的情况，详见llamafactory官方仓库的issue#6111，因此使用zero-2

单机修改
