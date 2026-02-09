# 说明

750b-distributed-train.yaml/sh为基于megatron框架，多机lora微调750B模型的配置文件、脚本。

40b-distributed-train.yaml/sh为基于llamafactory框架、集成muon优化器+deepspeed，用于**多机分布式**全量训练40B模型的配置文件、脚本，使用的镜像即为muon+deepspeed目录中代码构建好的镜像。

两个脚本的前半部分一致，包括配置环境变量、清理缓存、建立日志目录等环节，区别在于具体训练方式（二者所用框架不同）

# 注意

**在包头集群上，脚本中的文件的路径需要是：/mnt/workspace/wanghao277/... 而非/mnt/jpfs-5p/wanghao277/...**

# Megatron SFT 分布式训练脚本（LoRA）

**具体执行流程在第5.4节，README的其余部分为相关的说明**

750b-distributed-train.sh脚本用于在多节点多卡环境下基于megatron框架启动分布式训练，具体包含：

- NCCL/通信环境变量设置
- HuggingFace modules cache 清理与重新指定缓存目录
- 建立训练日志（每个节点独立日志文件，带时间戳）
- 进行 LoRA 微调

在本节中，配置文件与脚本指的是750b-distributed-train.yaml与750b-distributed-train.sh

## 1. 环境配置

### 1.1 环境依赖

配置文件第27行指定了使用的镜像，包含了ms-swift框架

### 1.2 权限要求

脚本第86-92行会尝试向 `/etc/hosts` 追加一行映射，需要容器/环境允许写 `/etc/hosts`

## 2. 日志与文件

按实际情况修改以下路径：

- 日志目录，脚本第25行：
  - `log_file_dir=...`
- 训练日志目录，脚本第32行：
  - `TRAIN_LOG_DIR=...`
- 输出目录，脚本第180行：
  - `--save ...`
- 模型权重，脚本第102行、第154行：
  - `DENSE_CKPT=...`
  - `--model ...`
- 数据集目录，脚本第111行：
  - `DATA_PATH=...`

## 3. 关键配置

### 3.1 通信与性能环境变量

脚本第1-7行设置了常见 NCCL 优化参数，根据实际情况进行更改：

- 必选：
  - `NCCL_SOCKET_IFNAME=eth0`
- 可选（提升通信性能）：
  - `NCCL_PXN_DISABLE=0`
  - `NCCL_CROSS_NIC=1`
  - `NCCL_IB_QPS_PER_CONNECTION=4`
- 其他：
  - `TORCH_NCCL_ENABLE_MONITORING=0`
  - `OMP_NUM_THREADS=1`
  - `WANDB_MODE=offline`

### 3.2 HF modules cache 清理与重定向

脚本第12-16行会强制清理 HuggingFace 动态模块缓存，避免旧的 remote code 版本造成bug，并新建新的缓存目录：

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/
rm -rf /root/.cache/huggingface/modules/transformers_modules/
sleep 2

export HF_MODULES_CACHE="/mnt/workspace/wanghao277/hf_cahce/hf_cache_temp_$(date +%s)"
mkdir -p $HF_MODULES_CACHE
```

## 4. 节点同步

脚本通过 RANK 的编号区分主/从 RANK

- RANK=0（主RANK）：
  脚本第27行，写入主节点 IP 到：`$log_file_dir/host_ip.txt`
  脚本第28行，写入主节点 hostname 到：`$log_file_dir/host_name.txt`
- RANK!=0（从RANK）：
  脚本第81-82行，等待上述两个文件出现，随后脚本会读取：
  `master_address=$(head -n 1 "$log_file")`
  `master_name=$(head -n 1 "$name_file")`

## 5.训练命令

### 5.1 设置分布式变量

脚本第106行：

- `NNODES=${WORLD_SIZE:-16}`

脚本第146-151行：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `NNODES=$NNODES`
- `NODE_RANK=$RANK`
- `MASTER_ADDR=$master_address`
- `MASTER_PORT=22`
- `NPROC_PER_NODE=8`

### 5.2 并行说明

脚本中启动训练的部分为第153-191行：

```bash
megatron sft \
        ...
        --tensor_model_parallel_size 8 \
        --sequence_parallel true \
        --expert_model_parallel_size 16 \
        --pipeline_model_parallel_size 1 \
        --context_parallel_size 4 \
        --moe_grouped_gemm true \
        --moe_shared_expert_overlap true \
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

脚本第94-100行注释掉的如下代码：

```bash
# log_info "开始执行MoE→Dense权重转换..."
# python ${MEGATRON_LM_PATH}/tools/convert_moe_to_dense.py \
#        --load /mnt/public/zhangtianyi/MCP/checkpoints/mcore/Meta-Llama-3.1-70B \
#        --save /mnt/public/zhangtianyi/MCP/checkpoints/mcore/Meta-Llama-3.1-70B-DENSE 2>&1 | tee -a "$FULL_LOG_PATH"
# log_info "权重转换完成！"
```

如果 checkpoint 是 MoE 格式，可取消注释执行一次转换，并在转换完成之后，在训练时将 DENSE_CKPT 指向转换后的 Dense 权重目录即可，目前已经做过转换，已经将这段注释掉并设置了 DENSE_CKPT 参数

### 5.4 启动训练步骤

使用脚本启动新的训练，所需要更改的位置不多，具体步骤如下：

1.修改脚本中两个日志目录的路径(或者删除旧的日志目录)：

- log_file_dir
- TRAIN_LOG_DIR

2.自定义分布式变量：

- NNODES：机器数（对应配置文件中的 replicas 参数）
- NPROC_PER_NODE：每台机器的显卡数量

3.修改脚本底部启动训练命令中的参数如：

- 模型路径：--model
- 数据集路径：--dataset
- 输出路径：--save
- 各个并行参数
- lora参数
- 其他参数

4.启动脚本：

```bash
k apply -f 750b-distributed-train.yaml
```

# LLaMA-Factory（Muon + DeepSpeed）分布式全量训练训练脚本

40b-distributed-train.sh脚本用于在多节点多卡环境下基于llamafactory框架并使用DeepSpeed与Muon优化器启动分布式全量训练，具体包含：

- NCCL/通信环境变量设置
- HuggingFace modules cache 清理与重新指定缓存目录
- 建立训练日志（每个节点独立日志文件，带时间戳）
- 通过 `FORCE_TORCHRUN=1` 使用 torchrun 分布式启动
- 使用 DeepSpeed 配置（ZeRO-2）与 Muon优化器

**脚本前面的部分与750b-distributed-train.sh类似，差异在于启动训练的部分**

在本节中，配置文件与脚本指的是40b-distributed-train.yaml与40b-distributed-train.sh

## 启动训练步骤

1.在脚本中第38行、45行修改日志目录或删除旧的日志目录：

- `log_file_dir=...`
- `TRAIN_LOG_DIR=...`

2.自定义分布式变量：

- NUM_NODES：机器数（对应配置文件中的 replicas 参数）
- GPUS_PER_NODE=8：每台机器的显卡数量

3.在脚本中第126行指定数据集文件路径：

```bash
TRAIN_FILE=...
```

4.配置dataset_info.json的内容

5.修改脚本底部第206-238行，启动训练命令中的参数：

- --dataset_dir参数为dataset_info.json所在的路径
- --dataset参数为在dataset_info.json中定义的数据集名称
- 其余的如模型路径、输出路径、学习率及epoch等

6.**修改要训练的模型的目录中modeling_deepseek.py的moe类的moe函数代码,具体修改方式参考muon+deepspeed文件夹下的多机代码修改过程.txt底部的内容**(我把40B模型的目录中的修改之后的modeling_deepseek.py拷贝在当前目录下，可以参考)

7.启动脚本：

```bash
k apply -f 40b-distributed-train.yaml
```

### 说明

脚本中第206-238行，启动训练的部分为：

```bash
FORCE_TORCHRUN=1 NNODES=${NUM_NODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
  llamafactory-cli train \
    --model_name_or_path ... \
    --dataset_dir ... \
    ...
```

1.使用torchrun命令启动分布式训练，符合llamafactory官方的启动方式

2.使用的deepspeed配置为zero-2，使用zero-3会存在训练速度很慢的情况，详见llamafactory官方仓库的issue#6111，因此使用zero-2

3.由于框架为llamafactory，因此需要配置dataset_info.json，配置dataset_info.json并将训练命令中的dataset_dir参数设置为dataset_info.json所在的路径

4.注意，在脚本第200行设置了DISABLE_VERSION_CHECK环境变量，不可取消：

```bash
export DISABLE_VERSION_CHECK=1 # 重要
```
