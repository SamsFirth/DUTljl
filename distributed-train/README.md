
### 说明

ds-train.yaml/sh为基于megatron框架，多机lora微调deepseek-v3模型的配置文件、脚本。

40b-distributed-train.yaml/sh为基于llamafactory框架，集成muon优化器+deepspeed并用于多机分布式全量训练40B模型的配置文件、脚本，其中使用的镜像即为使用muon+deepspeed目录中代码构建好的镜像。

注意ds-train.sh中的训练启动方式：

```bash

megatronsft\

  ...

--tensor_model_parallel_size8\

  --expert_model_parallel_size 16 \

--context_parallel_size4\

  --sequence_parallel true \

--pipeline_model_parallel_size1\

```

1.开启了张量并行、专家并行、上下文并行、序列并行

2.没有开启流水线并行

注意40b-distributed-train.sh中的训练启动方式：

```bash

FORCE_TORCHRUN=1NNODES=${NUM_NODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \

  llamafactory-cli train \

--model_name_or_path...\

    --dataset_dir ... \

...

```

1.使用torchrun命令启动分布式训练，符合llamafactory的启动方式

2.使用的deepspeed为zero-2，使用zero-3会存在训练速度很慢的情况，详见llamafactory官方仓库的issue#6111，因此使用zero-2
