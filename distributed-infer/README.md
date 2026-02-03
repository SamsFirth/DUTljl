
# SGLang 分布式部署脚本说明

sglang-infer.yaml/sh为在集群上基于sglang进行分布式推理的配置文件、脚本。用于在多机多卡环境中启动 **SGLang 推理服务**，并在 **RANK=0** 节点启动 **sglang-router**，将多个 worker 的服务地址聚合成一个路由入口

# 执行脚本
-`k apply -f sglang-infer.yaml`

## 1. 功能概览

脚本的主要功能：

1. 安装依赖
2. 设置 NCCL / GLOO / IB 等通信环境变量（需按集群适配）
3. 读取本机 IP，并写入 `$LOGS/tmp/ip_node_${RANK}.txt`
4. 每个 RANK 后台启动一个 worker：

-`python3 -m sglang.launch_server ... --port 30001`

5.`RANK=0` 收集每个 “实例组” 的首节点 IP（步长为 `NODE_PER_INSTANCE`）

6.`RANK=0` 启动 router，并把 `http://<ip>:30001` 作为 worker urls 传入

7.`sleep inf` 保持进程不退出

## 2. 目录与文件

- 日志目录：

`LOGS=/mnt/workspace/wanghao277/ljl_muon_infer_logs/$NAMES`（注意修改）

相关文件：

-`$LOGS/run${RANK}.log`：每个 rank 的 worker 日志

-`$LOGS/router.log`：router 日志（仅 RANK=0）

-`$LOGS/tmp/ip_node_${RANK}.txt`：每个 rank 写入的 IP

-`$LOGS/tmp/even_node_ips.txt`：RANK=0 收集到的 worker IP 列表

## 3. 环境依赖

### 3.1 必需依赖

脚本会执行：

-`pip install /mnt/workspace/wanghao277/packages/openai-1.76.2-py3-none-any.whl`（需要提前将wheel下载到本地，并更改命令的路径）

-`RANK=0` 时安装 `sglang-router`

### 3.2 设置相关的环境变量

在脚本上方，例如：

-`NCCL_TIMEOUT=1800`

-`GLOO_SOCKET_IFNAME=eth0`

-`NCCL_SOCKET_IFNAME=eth0`

-`NCCL_DEBUG=INFO`

## 4. 可选：使用function call模型（JoyAI 1.3T）

如果要部署 JoyAI 1.3T 的 function call 模型，需要先把 parser / detector 拷贝进 sglang 源码目录。

具体代码位于脚本顶部与启动命令中，已经被我注释，需要则取消注释即可：

```bash

# cp .../function_call_parser.py  /sgl-workspace/sglang/python/sglang/srt/function_call

# cp .../qwen3_coder_check_detector.py /sgl-workspace/sglang/python/sglang/srt/function_call

...

# --tool-call-parser qwen3_coder_check \

```

## 5. 启动方式

### 5.1 启动方式

在配置文件中，启动命令为：

```bash

-bash

-/mnt/workspace/wanghao277/ljl_infer.sh

-/mnt/workspace/wanghao277/ljl-muon-new-64k-output/checkpoint-242# 模型路径

-checkpoint-242# 模型名称

-unused# system role，当前为无效参数，会从tokenizer里面读取chat template

-"8"# tp

-"4"# pp

-"8"# ep

-"4"# 每个副本由节点/容器部署

-"4"# 共计有多少个节点/容器

```

### 5.2 脚本的参数

脚本支持设置8个默认参数：

| 参数 | 默认值 |

|---|---|

| INPUT_DIR | /mnt/workspace/wanghao277/hf/merge-with-s2-0528-19-dpo-LR3e-7-unification_dpo_0528_postive_origianl_negative_count25410 |

| NAMES | deepseek-v3-base-ddp |

| ROLE | chatrhino |

| TP | 32 |

| PP | 32 |

| EP | 32 |

| NODE_PER_INSTANCE | 4 |

| WORLD_SIZE | 16 |

在脚本中对应的启动服务的代码中，也可以自己设置这几个参数：

```bash

nohuppython3-msglang.launch_server\

        --model-path $INPUT_DIR \

--tp$TP\

        --pp-size $PP \

--ep$EP\

        --dist-init-addr $(cat $tmp_dir/ip_node_$(((RANK / $NODE_PER_INSTANCE) * $NODE_PER_INSTANCE )).txt):21000 \

--nnodes$NODE_PER_INSTANCE\

        --node-rank $((RANK % $NODE_PER_INSTANCE)) \

--host'0.0.0.0'\

        --port 30001 \

--mem-fraction-static0.9\

        --trust-remote-code \

--context-length20480\

        > $LOGS/run${RANK}.log 2>&1 &

# --cuda-graph-max-bs 16 \

# --disable-cuda-graph \

# --tool-call-parser qwen3_coder_check \

```

### 5.3 对外暴露的url

worker（每个节点/进程）：http://<worker_ip>:30001（注意端口是30001！！！）

router（仅 rank0 节点）：http://<rank0_ip>:30001

对应的ip需要在日志中查看（详见上面第二节，日志与文件）

## 6. 注意

- 每次启动脚本都要清空之前的日志，防止结构混乱
- 在脚本中若需要自定义最大上下文长度（默认读取模型config中的值），需要设置如下环境变量，以及在启动服务命令中添加自定义的context-length参数：

```bash

exportSGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

...

--context-length20480\

```

- 脚本中启动命令中，每行设置参数结尾的 **换行符** 后面不要有多余空格！！！
