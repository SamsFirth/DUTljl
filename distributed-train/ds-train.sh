export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SOCKET_IFNAME=eth0 # 必选
export NCCL_PXN_DISABLE=0 # 可选，提升通信性能
export NCCL_CROSS_NIC=1 # 可选，提升通信性能
export NCCL_IB_QPS_PER_CONNECTION=4 # 可选，提升通信性能
export OMP_NUM_THREADS=1  # 可选，torch默认参数
export WANDB_MODE=offline
# export MEGATRON_LM_PATH="/Megatron-LM"   # 你的 Megatron 源码目录
# export MODELSCOPE_CACHE="/shared"
mkdir -p /shared

# --- 新增：强制清理缓存代码 ---
# 无论是否共享存储，这行命令都能确保缓存被重置
rm -rf ~/.cache/huggingface/modules/transformers_modules/
# 或者更彻底一点（如果用的是默认路径）
rm -rf /root/.cache/huggingface/modules/transformers_modules/

# 稍微等待一下文件系统同步（如果是共享存储的话）
sleep 2

export HF_MODULES_CACHE="/mnt/workspace/wanghao277/hf_cahce/hf_cache_temp_$(date +%s)"
mkdir -p $HF_MODULES_CACHE

# conda_env="/usr/local"
log_file_dir='/mnt/workspace/wanghao277/ljl_3_logs' # 可改，每次重新训练要把log、train_log手动删除！！！！！！！！！！
mkdir -p "$log_file_dir"
log_file="$log_file_dir/host_ip.txt"
name_file="$log_file_dir/host_name.txt"

# -------------------- 新增：日志输出配置 --------------------
# 日志根目录（建议和训练输出分开，方便管理）
TRAIN_LOG_DIR="/mnt/workspace/wanghao277/ljl_3_train_logs" # 可改，每次重新训练要把log、train_log手动删除！！！！！！！！！！
mkdir -p "$TRAIN_LOG_DIR"  # 确保目录存在

# 日志文件名：包含节点Rank、时间戳，避免覆盖
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  # 时间戳：年-月-日_时-分-秒
LOG_FILENAME="node_rank_${RANK}_train_${TIMESTAMP}.log"
FULL_LOG_PATH="${TRAIN_LOG_DIR}/${LOG_FILENAME}"

# 定义日志函数：带时间戳输出，同时写入文件和终端
log_info() {
    local msg="$1"
    local time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$time] [INFO] [Node-$RANK] $msg" | tee -a "$FULL_LOG_PATH"
}

log_warn() {
    local msg="$1"
    local time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$time] [WARN] [Node-$RANK] $msg" | tee -a "$FULL_LOG_PATH"
}

log_error() {
    local msg="$1"
    local time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$time] [ERROR] [Node-$RANK] $msg" | tee -a "$FULL_LOG_PATH"
}

# 初始化日志文件（写入脚本启动信息）
log_info "==================================== 训练脚本启动 ===================================="
log_info "日志文件路径：$FULL_LOG_PATH"
log_info "节点Rank：$RANK"
log_info "WANDB模式：$WANDB_MODE"
log_info "Megatron-LM路径：$MEGATRON_LM_PATH"

# -------------------- 1. 主节点发现 --------------------
if [ "${RANK}" -eq 0 ]; then
    NODE_TYPE=--head
    log_info "当前节点为：主节点（Head Node）"
    hostname -I | awk '{print $1}' > "${log_file}"
    hostname > "${name_file}"
else
    NODE_TYPE=--worker
    log_info "当前节点为：工作节点（Worker Node）"
    log_info "等待主节点IP文件（$log_file）..."
    while [ ! -f "$log_file" ]; do sleep 5; done
    log_info "等待主节点名称文件（$name_file）..."
    while [ ! -f "$name_file" ]; do sleep 5; done
fi

master_address=$(head -n 1 "$log_file")
master_name=$(head -n 1 "$name_file")
log_info "主节点IP：$master_address"
log_info "主节点名称：$master_name"

# 写 /etc/hosts（仅一次）
if grep -q "$master_address $master_name" /etc/hosts; then
    log_info "/etc/hosts已存在主节点映射：$master_address $master_name"
else
    log_info "向/etc/hosts添加主节点映射：$master_address $master_name"
    echo "$master_address $master_name" | tee -a /etc/hosts > /dev/null
fi

# -------------------- 2. 可选：MoE→Dense 权重转换 --------------------
# 如果你的 checkpoint 还是 MoE，请先把下面注释打开，跑完一次即可
# log_info "开始执行MoE→Dense权重转换..."
# python ${MEGATRON_LM_PATH}/tools/convert_moe_to_dense.py \
#        --load /mnt/public/zhangtianyi/MCP/checkpoints/mcore/Meta-Llama-3.1-70B \
#        --save /mnt/public/zhangtianyi/MCP/checkpoints/mcore/Meta-Llama-3.1-70B-DENSE 2>&1 | tee -a "$FULL_LOG_PATH"
# log_info "权重转换完成！"

DENSE_CKPT="/mnt/workspace/wanghao277/merge-with-s2-0528-19-dpo-LR3e-7-unification_dpo_0528_postive_origianl_negative_count25410"  # 已转换好的 Dense 权重
log_info "使用的Dense权重路径：$DENSE_CKPT"

# -------------------- 3. 训练参数计算 --------------------
export NNODES=${WORLD_SIZE:-16}
GLOBAL_BATCH=64
# DATA_PATH=/mnt/llm-train/users/explore-train/wangzhenfang8/datasets/wanghao/train
# DATA_PATH=/mnt/llm-train/users/explore-train/wangzhenfang8/datasets/wanghao/train_16k
# DATA_PATH=/mnt/llm-train/users/explore-train/wangzhenfang8/datasets/wanghao/train_4k
DATA_PATH=/mnt/workspace/wanghao277/data/0108/alpaca/
log_info "数据路径：$DATA_PATH"

# 统计数据行数（带日志输出）
if [ $(ls -1 "$DATA_PATH"/*.jsonl 2>/dev/null | wc -l) -eq 0 ]; then
    log_error "错误：$DATA_PATH 目录下没有找到jsonl文件！"
    exit 1
fi
COUNT=$(cat "$DATA_PATH"/*.jsonl | wc -l)
log_info "数据总行数：$COUNT"

TRAINING_STEPS=$(($COUNT / $GLOBAL_BATCH))
DECAY_STEPS=$(($TRAINING_STEPS * 99 / 100))
SAVE_INTERVAL=$(($COUNT / $GLOBAL_BATCH))
WARMUP_STEPS=$(($TRAINING_STEPS-$DECAY_STEPS))

# 打印所有关键训练参数（写入日志）
log_info "==================================== 训练参数汇总 ===================================="
log_info "节点总数（NNODES）：$NNODES"
log_info "全局批次大小（GLOBAL_BATCH）：$GLOBAL_BATCH"
log_info "训练总步数（TRAINING_STEPS）：$TRAINING_STEPS"
log_info "预热步数（WARMUP_STEPS）：$WARMUP_STEPS"
log_info "衰减步数（DECAY_STEPS）：$DECAY_STEPS"
log_info "模型保存间隔（SAVE_INTERVAL）：$SAVE_INTERVAL"
log_info "最大epoch（max_epochs）：1"
log_info "最大序列长度（max_length）：65536"
log_info "学习率（lr）：3e-6"
log_info "模型加载路径（--load）：$DENSE_CKPT"
log_info "======================================================================================"

# -------------------- 4. 拉起训练（日志定向输出） --------------------
log_info "开始启动分布式训练..."
# log_info "训练命令：${conda_env}/bin/python3 -m torch.distributed.run ..."

# 关键：将训练的stdout和stderr都写入日志文件，同时保留终端输出（2>&1 | tee -a）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=$NNODES
export NODE_RANK=$RANK
export MASTER_ADDR=$master_address
export MASTER_PORT=22
export NPROC_PER_NODE=8

megatron sft \
        --model /mnt/workspace/wanghao277/hf/merge-with-s2-0528-19-dpo-LR3e-7-unification_dpo_0528_postive_origianl_negative_count25410 \
        --load ${DENSE_CKPT} \
        --dataset ${DATA_PATH} \
        --model_type deepseek_v2_5 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --train_type lora \
        --target_modules all-linear \
        --tensor_model_parallel_size 8 \
        --sequence_parallel true \
        --expert_model_parallel_size 16 \
        --pipeline_model_parallel_size 1 \
        --optimizer_cpu_offload true \
        --use-precision-aware-optimizer \
        --save_interval 50 \
        --context_parallel_size 4 \
        --moe_grouped_gemm true \
        --moe_shared_expert_overlap true \
        --moe_aux_loss_coeff 0.01 \
        --micro_batch_size 1 \
        --global_batch_size ${GLOBAL_BATCH} \
        --max_epochs 1 \
        --max_length 20480 \
        --lr 1e-6 \
        --lr_warmup_iters 10 \
        --min_lr 0 \
        --save /mnt/workspace/wanghao277/ljl_3_outputs \
        --num_workers 4 \
        --dataset_num_proc 8 \
        --recompute_granularity full \
        --recompute_method uniform \
        --recompute_num_layers 1 \
        --no_save_optim true \
        --no_save_rng true \
        --finetune true \
        --cross_entropy_loss_fusion true \
        --use_flash_attn true \
        --loss_scale ignore_empty_think 2>&1 | tee -a "$FULL_LOG_PATH"

# 训练结束标记
if [ $? -eq 0 ]; then
    log_info "==================================== 训练正常结束 ===================================="
else
    log_error "==================================== 训练异常退出 ===================================="
    exit 1
fi
