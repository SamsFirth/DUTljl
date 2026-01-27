### 注意事项
在本仓库中，joyaiSFT文件夹与jdea文件夹中是两个包的源码，二者对应的whl包已经上传到网页的发行版中（下面的环境配置教程中提到的这两个包与此处发行版网页中的是一样的）

**注意** 
需要被加密的文件有两个：
jdea/src/jdea/hparams/model_config_inline.py
jdea/src/jdea/hparams/optimize_rules_inline.py
**注意**

### 环境安装
1. 新建虚拟环境，注意python版本是 **3.11** ：
```bash
conda create -n your_env_name python=3.11
```
2. 激活虚拟环境，并执行如下两条命令：
```bash
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime
```
3. 克隆仓库：
```bash
git clone git@coding.jd.com:wanghao277/lowresource-sft.git
```
4. 进入仓库：
```bash
cd lowresource-sft
```
5. 安装torch：
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```
6. 执行如下命令，获取joyaisft包，注意将第一个参数 **`<api-token>`** 替换为自己的api-token（[API Token 生成入口](http://xingyun.jd.com/codingRoot/profile/settings/apiToken)）：
```bash
wget \
  --header="PRIVATE-TOKEN: <api-token>" \
  -O joyaisft-1.0-cp311-cp311-linux_x86_64.whl \
  "https://coding.jd.com/webapi/wanghao277/lowresource-sft/files/30819/joyaisft.1.0.cp311.cp311.linux.x86.64.whl"
```
7. 安装joyaisft与环境依赖：
```bash
pip install joyaisft-1.0-cp311-cp311-linux_x86_64.whl
```
8. 执行如下命令，获取jdea包，注意替换 **`<api-token>`**：
```bash
wget \
  --header="PRIVATE-TOKEN: <api-token>" \
  -O jdea-1.0-py3-none-any.whl \
  "https://coding.jd.com/webapi/JDEA-ChatRhino/lowsource-sft/files/30249/jdea.1.0.py3.none.any.whl"
```
9. 安装jdea：
```bash
pip install jdea-1.0-py3-none-any.whl
```
10. （flash-attn包已经在该仓库的发行版网页中，可以直接下载，不必按照下面的命令下载）执行如下命令，获取flash-attention包，注意替换 **`<api-token>`**：
```bash
wget \
  --header="PRIVATE-TOKEN: <api-token>" \
  -O flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  "https://coding.jd.com/webapi/JDEA-ChatRhino/lowsource-sft/files/29823/flash.attn.2.8.3.cu12torch2.8cxx11abiTRUE.cp311.cp311.linux.x86.64.whl"
```
11. 安装flash-attention：
```bash
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```
12. 环境安装完毕

### 微调
1. 配置数据集的配置文件，位置在：lowresource-sft/run/data/dataset_info.json
注意数据集要求为如下的json格式：
```json
[
  {
    "instruction": “XXX”,
    "output": “XXX”
  }
]
```
2. 配置lora微调的配置文件，位置在：lowresource-sft/run/750B-sft-lora.yaml
3. 在lowresource-sft/run/路径下，执行如下命令，开始微调模型：
```bash
python train.py 750B-sft-lora.yaml
```

### 与微调后模型交互
1. 配置推理的配置文件，位置在：lowresource-sft/run/750B-sft-infer.yaml
2. 在lowresource-sft/run/路径下，执行如下命令，加载lora的adapter并与模型进行对话：
```bash
python chat.py 750B-sft-infer.yaml
```

### API生成
1. 配置推理的配置文件，位置在：lowresource-sft/run/750B-sft-infer.yaml
2. 在lowresource-sft/run/路径下，执行如下命令，生成微调后模型的API：
```bash
API_PORT=8000 python api.py 750B-sft-infer.yaml
```