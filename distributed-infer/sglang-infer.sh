#!/bin/bash
set -x
# cp /mnt/llm-train/users/explore-train/zhangtianyi39/MCP/sglang_utils/0.5.6/function_call_parser.py /sgl-workspace/sglang/python/sglang/srt/function_call
# cp /mnt/llm-train/users/explore-train/zhangtianyi39/MCP/sglang_utils/0.5.6/qwen3_coder_check_detector.py /sgl-workspace/sglang/python/sglang/srt/function_call
# 如果需要部署joyai 1.3T function call模型，则需要以上两个步骤

pip install /mnt/workspace/wanghao277/packages/openai-1.76.2-py3-none-any.whl
# pip install /mnt/public/wangzhenfang8/whl/sglang_router-0.1.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
if [ "$RANK" -eq 0 ]; then
    pip install sglang-router -i https://mirrors.jd.com/pypi/web/simple
fi

export NCCL_TIMEOUT=1800      # 设置 NCCL 超时为 30 分钟
export TORCH_DIST_TIMEOUT=1800
export UNBALANCED_MODEL_LOADING_TIMEOUT_S=1800

######################帅哥的神秘变量#####################
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=SLOT0:1,SLOT1:1,SLOT2:1,SLOT3:1,SLOT4:1,SLOT5:1,SLOT6:1,SLOT7:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_TC=96
export NCCL_IB_TIMEOUT=20
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IBEXT_DISABLE=0
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_DEBUG=WARN
export MELLANOX_VISIBLE_DEVICES=all
######################################################

###################train脚本的环境变量###################
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export SGL_ENABLE_JIT_DEEPGEMM=0
# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export NVTE_FUSED_ATTN=1
# export CUDNN_LOGERR_DBG=1
# export CUDNN_LOGDEST_DBG=stderr

# export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_SOCKET_IFNAME=eth0 # 必选
# export NCCL_PXN_DISABLE=0 # 可选，提升通信性能
# export NCCL_CROSS_NIC=1 # 可选，提升通信性能
# export NCCL_IB_QPS_PER_CONNECTION=4 # 可选，提升通信性能
# export OMP_NUM_THREADS=1  # 可选，torch默认参数
# # export NCCL_PXN_DISABLE=0 # 可选，提升通信性能
# # export NCCL_CROSS_NIC=1 # 可选，提升通信性能
# # export NCCL_IB_QPS_PER_CONNECTION=4 # 可选，提升通信性能
# # export OMP_NUM_THREADS=1  # 可选，torch默认参数

# # Distributed training variables
# # export NNODES=${WORLD_SIZE:-1}  # 以下均为适配平台环境变量
# export NNODES=64
# export GPUS_PER_NODE=8
# export GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
# export WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
# export NODE_RANK=${RANK:-0}
# export MASTER_ADDR=${MASTER_ADDR:-service-train-v3-s1-0}
# export MASTER_PORT=${MASTER_PORT:-22}
# export DYNAMIC_LENGTH=1

# source /mnt/llm-train/codes/megatron-lm-recipes/my.cfg # HOME_PATH and WANDB key
# export WANDB_MODE=offline
# # wandb login ${WANDB}
###############################################################
## 以上环境变量，在使用不同集群时各不相同，请参考训练脚本

cd /mnt/workspace/wanghao277
INPUT_DIR=${1-"/mnt/workspace/wanghao277/hf/merge-with-s2-0528-19-dpo-LR3e-7-unification_dpo_0528_postive_origianl_negative_count25410"}
NAMES=${2-"deepseek-v3-base-ddp"}
ROLE=${3-"chatrhino"}
TP=${4-32}
PP=${5-32}
EP=${6-32}
NODE_PER_INSTANCE=${7-4}
WORLD_SIZE=${8-16}
LOGS=/mnt/workspace/wanghao277/ljl_muon_infer_logs/$NAMES
# 这个LOGS的位置很重要，要改成自己的，每次模型部署的时候会把ip写到tmp中，然后会在第一个节点启动router，访问第一个节点的30000端口即可

mkdir -p $LOGS

TOTAL_NODES=$WORLD_SIZE

tmp_dir=$LOGS/tmp
mkdir -p $tmp_dir

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 # 强制扩大上下文长度，用后即删！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

MY_IP=$(cat /etc/hosts | grep $(hostname) | awk '{print $1}' | head -n 1)
echo "Node $RANK IP: $MY_IP"
echo "$MY_IP" > $tmp_dir/ip_node_${RANK}.txt
sleep 120 #确保所有的都已写入
nohup python3 -m sglang.launch_server \
        --model-path $INPUT_DIR \
        --tp $TP \
        --pp-size $PP \
        --ep $EP \
        --dist-init-addr $(cat $tmp_dir/ip_node_$(((RANK / $NODE_PER_INSTANCE) * $NODE_PER_INSTANCE )).txt):21000 \
        --nnodes $NODE_PER_INSTANCE \
        --node-rank $((RANK % $NODE_PER_INSTANCE)) \
        --host '0.0.0.0' \
        --port 30001 \
        --mem-fraction-static 0.9 \
        --trust-remote-code \
        --context-length 20480 \
        > $LOGS/run${RANK}.log 2>&1 &
        # --context-length 20480 \
        # --cuda-graph-max-bs 16 \
        # --disable-cuda-graph \
        # --tool-call-parser qwen3_coder_check \

if [ "$RANK" -eq 0 ]; then

    echo "Collecting even-numbered node IPs..."
    > $tmp_dir/even_node_ips.txt
    for i in $(seq 0 $NODE_PER_INSTANCE $(($TOTAL_NODES-1))); do
        if [ -f "$tmp_dir/ip_node_${i}.txt" ]; then
            NODE_IP=$(cat $tmp_dir/ip_node_${i}.txt)
            echo "${NODE_IP}" >> $tmp_dir/even_node_ips.txt
        else
            echo "Warning: IP file for even node $i not found!"
        fi
    done
    echo "Even-numbered node IPs collected in $tmp_dir/even_node_ips.txt"
fi

if [ "$RANK" -eq 0 ]; then
    ips=$(cat $tmp_dir/even_node_ips.txt)
    cmd="python3 -m sglang_router.launch_router --worker-urls"
    for ip in $ips; do
        cmd="$cmd http://$ip:30001"
    done
fi
# 启动router

if [ "$RANK" -eq 0 ]; then

    MODEL_PATH=$INPUT_DIR
    URL=0.0.0.0
    while ! curl -s --head --request POST "http://$URL:30001/v1/completions" | grep "server: uvicorn" > /dev/null; do
        echo "无法连接到$URL，正在重试..."
        sleep 5 # 等待 5 秒再重试
    done
    sleep 60
    echo "http://$URL:30001/v1/completions 可用，执行router"
    nohup $cmd --host $URL --balance-abs-threshold 4 > $LOGS/router.log 2>&1 &
    sleep 10
fi
# echo "command:"
# echo "bash /mnt/public/wuwen_data/codes/opencompass_v2_vllm/run_sglang.sh $MODEL_PATH $NAMES mmlu_ppl http://$MY_IP:30000/v1/chat/completions"

sleep inf