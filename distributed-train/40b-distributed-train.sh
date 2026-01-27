export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SOCKET_IFNAME=eth0 # 必选
export NCCL_PXN_DISABLE=0 # 可选，提升通信性能
export NCCL_CROSS_NIC=1 # 可选，提升通信性能
export NCCL_IB_QPS_PER_CONNECTION=4 # 可选，提升通信性能
export OMP_NUM_THREADS=1  # 可选，torch默认参数
export WANDB_MODE=offline

# # ---------- RDMA/NCCL 诊断与选择 ----------
# export NCCL_DEBUG=INFO  # 后添加的


# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_DEBUG_SUBSYS=INIT,NET

export NCCL_TIMEOUT=7200
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  #测试！！！！！！！！！！！！！！！！
export TORCH_DISTRIBUTED_DEBUG=DETAIL

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
log_file_dir='/mnt/workspace/wanghao277/ljl_muon_99999_test_logs' # 可改，每次重新训练要把log、train_log手动删除！！！！！！！！！！
mkdir -p "$log_file_dir"
log_file="$log_file_dir/host_ip.txt"
name_file="$log_file_dir/host_name.txt"

# -------------------- 新增：日志输出配置 --------------------
# 日志根目录（建议和训练输出分开，方便管理）
TRAIN_LOG_DIR="/mnt/workspace/wanghao277/ljl_muon_99999_test_train_logs" # 可改，每次重新训练要把log、train_log手动删除！！！！！！！！！！
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
# log_info "Megatron-LM路径：$MEGATRON_LM_PATH"

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

# -------------------- 3. 训练参数计算 --------------------
GLOBAL_BATCH=64

DATA_PATH=/mnt/workspace/wanghao277/data/alpaca/
log_info "数据路径：$DATA_PATH"

# 检查数据文件是否存在（json/jsonl 都支持）
if [ $(find "$DATA_PATH" -maxdepth 1 -type f \( -name "*.jsonl" -o -name "*.json" \) | wc -l) -eq 0 ]; then
    log_error "错误：$DATA_PATH 目录下没有找到 jsonl 或 json 文件！"
    log_error "DEBUG: pwd=$(pwd)"
    log_error "DEBUG: list DATA_PATH:"
    ls -la "$DATA_PATH" | head -200 | tee -a "$FULL_LOG_PATH"
    log_error "DEBUG: find json/jsonl:"
    find "$DATA_PATH" -maxdepth 2 -type f \( -name "*.jsonl" -o -name "*.json" \) | head -200 | tee -a "$FULL_LOG_PATH"
    log_error "进入排错保持状态：sleep infinity（便于 kubectl exec）"
    sleep infinity
fi

# 指定训练集文件（你现在是 .json）
TRAIN_FILE="$DATA_PATH/wmzy_train.json" #这里！！！！！！！！！！！！！ 
if [ ! -f "$TRAIN_FILE" ]; then
    log_error "错误：训练集文件不存在：$TRAIN_FILE"
    ls -la "$DATA_PATH" | head -200 | tee -a "$FULL_LOG_PATH"
    sleep infinity
fi

# 统计样本数：
# - jsonl：按非空行数
# - json（alpaca 常见）：按顶层数组长度
COUNT=$(TRAIN_FILE="$TRAIN_FILE" python3 - << 'PY'
import os, json, sys

path = os.environ["TRAIN_FILE"]
if path.endswith(".jsonl"):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    print(n)
    sys.exit(0)

# json：通常是 list[dict]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)

if isinstance(obj, list):
    print(len(obj))
    sys.exit(0)

# 少数情况下是 dict 包 list（给个明确报错，便于你立刻定位结构）
raise SystemExit(f"Unexpected JSON schema: top-level type={type(obj)}; keys={list(obj)[:20] if isinstance(obj, dict) else ''}")
PY
)

log_info "数据总样本数：$COUNT"

TRAINING_STEPS=$(($COUNT / $GLOBAL_BATCH))
DECAY_STEPS=$(($TRAINING_STEPS * 99 / 100))
SAVE_INTERVAL=$(($COUNT / $GLOBAL_BATCH))
WARMUP_STEPS=$(($TRAINING_STEPS-$DECAY_STEPS))

# 可选：不写也行，k8s 一般每 pod 就是 0-7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NUM_NODES=16               # 你的 TrainingJob replicas
export GPUS_PER_NODE=8            # 每个 Pod 8 卡

# RANK 在你脚本里用于区分 head/worker，应该是 0..15 的“节点序号”
export NODE_RANK=${RANK}

export MASTER_ADDR=$master_address
export MASTER_PORT=22 # 这里！！！！！！！！！！！！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

log_info "torchrun: nnodes=${NUM_NODES}, nproc_per_node=${GPUS_PER_NODE}, node_rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}"

# 打印所有关键训练参数（写入日志）
log_info "==================================== 训练参数汇总 ===================================="
log_info "节点总数（NNODES）：$NNODES"
log_info "全局批次大小（GLOBAL_BATCH）：$GLOBAL_BATCH"
log_info "训练总步数（TRAINING_STEPS）：$TRAINING_STEPS"
log_info "预热步数（WARMUP_STEPS）：$WARMUP_STEPS"
log_info "衰减步数（DECAY_STEPS）：$DECAY_STEPS"
log_info "模型保存间隔（SAVE_INTERVAL）：$SAVE_INTERVAL"
# log_info "最大epoch（max_epochs）：1"
# log_info "最大序列长度（max_length）：65536"
# log_info "学习率（lr）：3e-6"
# log_info "模型加载路径（--load）：$DENSE_CKPT"
log_info "======================================================================================"

# -------------------- 4. 拉起训练（日志定向输出） --------------------
log_info "开始启动分布式训练..."

export DISABLE_VERSION_CHECK=1 # 重要
# 测试：
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=NVL

FORCE_TORCHRUN=1 NNODES=${NUM_NODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
        llamafactory-cli train \
        --model_name_or_path /mnt/workspace/wanghao277/sft-pt_64k-l2s1-lr1e-5 \
        --dataset_dir /mnt/workspace/wanghao277/data/ \
        --trust_remote_code True \
        --stage sft \
        --do_train True \
        --do_eval False \
        --finetuning_type full \
        --use_muon True \
        --deepspeed /mnt/workspace/wanghao277/deepspeed/ds_z2_config.json \
        --dataset wmzy_train \
        --template deepseek3 \
        --cutoff_len 20480 \
        --max_samples 100000 \
        --overwrite_cache True \
        --preprocessing_num_workers 128 \
        --dataloader_num_workers 4 \
        --output_dir /mnt/workspace/wanghao277/ljl-muon-99999-output \
        --logging_steps 20 \
        --save_steps 400 \
        --plot_loss True \
        --overwrite_output_dir False \
        --save_only_model True \
        --report_to none  \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1.0e-5 \
        --num_train_epochs 10000000.0 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --ddp_timeout 180000000 \
        --bf16 True 2>&1 | tee -a "$FULL_LOG_PATH"
        
# 训练结束标记
if [ $? -eq 0 ]; then
    log_info "==================================== 训练正常结束 ===================================="
else
    log_error "==================================== 训练异常退出 ===================================="
    exit 1
fi
