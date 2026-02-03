
# 说明

ds-train.yaml/sh为基于megatron框架，多机lora微调deepseek-v3模型的配置文件、脚本。

40b-distributed-train.yaml/sh为基于llamafactory框架、集成muon优化器+deepspeed，用于**多机分布式**全量训练40B模型的配置文件、脚本，使用的镜像即为muon+deepspeed目录中代码构建好的镜像。

两个脚本的前半部分一致，包括配置环境变量、清理缓存、建立日志目录等环节，区别在于具体训练方式（二者所用框架不同）

# 执行脚本

-进行多机lora微调：`k apply -f ds-train.yaml`

-进行多机muon+deepspeed全量训练：`k apply -f 40b-distributed-train.yaml`

# 注意

**在进行多机+muon+deepspeed全量训练时，参考muon+deepspeed文件夹下的多机代码修改过程.txt底部的内容，需要修改模型文件夹中modeling_deepseek.py的moe类的moe函数代码！！！**

# Megatron SFT 分布式训练脚本（LoRA）

ds-train.sh脚本用于在多节点多卡环境下基于megatron框架启动分布式训练，具体包含：

- NCCL/通信环境变量设置
- HuggingFace modules cache 清理与重新指定缓存目录
- 建立训练日志（每个节点独立日志文件，带时间戳）
- 进行 LoRA 微调

## 1. 环境配置

### 1.1 环境依赖

配置文件中指定了使用的镜像，包含了ms-swift框架

### 1.2 权限要求

脚本会尝试向 `/etc/hosts` 追加一行映射，需要容器/环境允许写 `/etc/hosts`

## 2. 日志与文件

按实际情况修改以下路径：

- 日志目录：

-`log_file_dir=/mnt/workspace/wanghao277/ljl_3_logs`

-`TRAIN_LOG_DIR=/mnt/workspace/wanghao277/ljl_3_train_logs`

- 输出目录：

-`--save /mnt/workspace/wanghao277/ljl_3_outputs`

- 模型权重：

-`DENSE_CKPT=...`

-`--model ...`

- 数据集目录：

-`DATA_PATH=/mnt/workspace/wanghao277/data/0108/alpaca/`

## 3. 关键配置

### 3.1 通信与性能环境变量

脚本设置了常见 NCCL 优化参数，根据实际情况进行更改：

- 必选：

-`NCCL_SOCKET_IFNAME=eth0`

- 可选（提升通信性能）：

-`NCCL_PXN_DISABLE=0`

-`NCCL_CROSS_NIC=1`

-`NCCL_IB_QPS_PER_CONNECTION=4`

- 其他：

-`TORCH_NCCL_ENABLE_MONITORING=0`

-`OMP_NUM_THREADS=1`

-`WANDB_MODE=offline`

### 3.2 HF modules cache 清理与重定向

脚本会强制清理 HuggingFace 动态模块缓存，避免旧的 remote code 版本造成bug，并新建新的缓存目录：

```bash

rm-rf~/.cache/huggingface/modules/transformers_modules/

rm-rf/root/.cache/huggingface/modules/transformers_modules/

sleep2


exportHF_MODULES_CACHE="/mnt/workspace/wanghao277/hf_cahce/hf_cache_temp_$(date +%s)"

mkdir-p$HF_MODULES_CACHE

```

## 4. 节点同步

脚本通过 RANK 的编号区分主/从 RANK

- RANK=0（主RANK）：

写入主节点 IP 到：`$log_file_dir/host_ip.txt`

写入主节点 hostname 到：`$log_file_dir/host_name.txt`

- RANK!=0（从RANK）：

等待上述两个文件出现，随后脚本会读取：

`master_address=$(head -n 1 "$log_file")`

`master_name=$(head -n 1 "$name_file")`

## 5.训练命令

### 5.1 设置分布式变量

-`NNODES=${WORLD_SIZE:-16}`

-`NODE_RANK=$RANK`

-`MASTER_ADDR=$master_address`

-`MASTER_PORT=22`

-`NPROC_PER_NODE=8`

-`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

### 5.2 启动训练

```bash

megatronsft\

        ...

--tensor_model_parallel_size8\

        --sequence_parallel true \

--expert_model_parallel_size16\

        --pipeline_model_parallel_size 1 \

--context_parallel_size4\

        --moe_grouped_gemm true \

--moe_shared_expert_overlaptrue\

        --moe_aux_loss_coeff 0.01 \

...

```

训练命令同时启用了多种并行方式来提升吞吐并适配超大模型：

- Tensor Parallel（TP）：张量并行
- Pipeline Parallel（PP）：流水并行
- Expert Parallel（EP）：专家并行
- Sequence Parallel：序列并行
- Context Parallel（CP）：上下文并行
- Data Parallel（DP）：数据并行

**注意**：脚本启动命令中，每行参数末尾的 **换行符** 后面不要有多余空格！！！

### 5.3 MoE-Dense权重转换

脚本中注释掉的如下代码：

```bash

# log_info "开始执行MoE→Dense权重转换..."

# python ${MEGATRON_LM_PATH}/tools/convert_moe_to_dense.py \

#        --load /mnt/public/zhangtianyi/MCP/checkpoints/mcore/Meta-Llama-3.1-70B \

#        --save /mnt/public/zhangtianyi/MCP/checkpoints/mcore/Meta-Llama-3.1-70B-DENSE 2>&1 | tee -a "$FULL_LOG_PATH"

# log_info "权重转换完成！"

```

如果 checkpoint 是 MoE 格式，可取消注释执行一次转换，并在转换完成之后，在训练时将 DENSE_CKPT 指向转换后的 Dense 权重目录

# LLaMA-Factory（Muon + DeepSpeed）分布式全量训练训练脚本

40b-distributed-train.sh脚本用于在多节点多卡环境下基于llamafactory框架并使用DeepSpeed与Muon优化器启动分布式全量训练，具体包含：

- NCCL/通信环境变量设置
- HuggingFace modules cache 清理与重新指定缓存目录
- 建立训练日志（每个节点独立日志文件，带时间戳）
- 通过 `FORCE_TORCHRUN=1` 强制走 torchrun 分布式启动
- 使用 DeepSpeed 配置（ZeRO-2）与 MuOn优化器

**脚本前面的部分与ds-train.sh类似，差异在于启动训练的部分**

启动训练命令：

```bash

FORCE_TORCHRUN=1NNODES=${NUM_NODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \

  llamafactory-cli train \

--model_name_or_path...\

    --dataset_dir ... \

...

```

**注意**：

1.使用torchrun命令启动分布式训练，符合llamafactory官方的启动方式

2.使用的deepspeed配置为zero-2，使用zero-3会存在训练速度很慢的情况，详见llamafactory官方仓库的issue#6111，因此使用zero-2

3.由于框架为llamafactory，因此需要配置dataset_info.json，将训练命令中的dataset_dir参数设置为dataset_info.json所在的文件夹

4.注意，在脚本中设置了DISABLE_VERSION_CHECK环境变量，不可取消：

```bash

exportDISABLE_VERSION_CHECK=1# 重要

```
