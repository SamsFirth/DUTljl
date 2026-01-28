### 说明
sglang-infer.yaml/sh在集群上基于sglang进行分布式推理的配置文件、脚本。

脚本用于在多节点环境中部署 SGLang 推理服务：
每个节点启动一个 sglang.launch_server Worker（监听 30001）
Rank0 收集“每个实例组的首节点 IP” 并启动 sglang_router.launch_router，把请求转发/均衡到各 Worker
最终的对外入口ip为ip_node_0.txt中的ip xxx，使用的方式为：http://xxx:30001/v1/chat/completions