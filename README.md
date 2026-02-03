
# JoyAI 超大模型后训练框架

<h1align="center">JoyAI 训练千亿/万亿大模型计算框架`</h1>`

## 说明

jdea目录与joyaiSFT目录中是两个包的源码，二者对应的whl包已经上传到网页的发行版中，用于单机微调750B模型，具体使用说明详见其中的README，按照其配置完环境后，通过仓库中run目录中代码进行训练、推理、生成api；

muon+deepspeed目录中的代码为基于llamafactory，集成muon优化器与deepspeed框架，可用于 **单机** 全量训练40B模型，并且对应镜像也已经上传，**多机**代码在distributed-train目录中；

distributed-train目录下为在集群上进行多机分布式训练的配置文件、脚本，包括对deepseek v3模型进行lora微调、使用muon+deepspeed全量训练40B模型;

distributed-infer目录下为在集群上基于sglang进行分布式推理的配置文件、脚本，与进行推理测试的代码。
