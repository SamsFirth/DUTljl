from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import Trainer
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled, is_torch_xpu_available, is_torch_mlu_available, is_torch_musa_available, is_torch_npu_available, is_torch_mps_available, is_torch_hpu_available, is_accelerate_available, is_apex_available, logging
from packaging import version
import os
import inspect
import functools
from typing import Union, Any, Dict, List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, TaskType
from datasets import Dataset
from torchviz import make_dot
from tqdm import tqdm
import os, json
from pathlib import Path
from accelerate import Accelerator
if is_accelerate_available('0.28.0'):
    from accelerate.utils import DataLoaderConfiguration
from accelerate import __version__ as accelerate_version
if version.parse(accelerate_version) > version.parse('1.3.0'):
    from accelerate.utils import TorchTensorParallelPlugin
if is_sagemaker_mp_enabled():
    from transformers.trainer_utils import smp_forward_backward
from joyaiSFT.sft.peft_utils.mapping import get_peft_model
logger = logging.get_logger(__name__)
from transformers.trainer_callback import TrainerState
from transformers.trainer import TRAINER_STATE_NAME, OPTIMIZER_NAME, SCHEDULER_NAME
from transformers.trainer_pt_utils import torch_distributed_zero_first
_RNG_CANDIDATES = ["rng_state.pth", "rng_state.pt", "rng_state.pkl"]
import time
import json
import shutil
from safetensors.torch import save_file as safe_save_file
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME  = "adapter_config.json"

import collections
import gc
import ctypes
# 加载 libc 用于手动触发内存归还
try:
    libc = ctypes.CDLL("libc.so.6")
except:
    libc = None

def flush_memory():
    import gc
    gc.collect() # 清除 Python 层的循环引用
    if libc:
        libc.malloc_trim(0) # 强制清理 C++ / glibc 的堆内存碎片

def _to_adapter_ckpt_key(model_param_name: str) -> str:
    # 反向适配你 load_joy_peft_model 里的映射：
    # checkpoint key: "base_model.model.<xxx>.weight"
    # model param:     "<xxx>.default.weight"
    name = model_param_name
    # 把 ".default." 去掉（核心）
    name = name.replace(".default.", ".")
    # 兼容末尾 ".default.weight"
    name = name.replace(".default.weight", ".weight")
    return "base_model.model." + name

class JOYAccelerator(Accelerator):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('device_placement', False)
        super().__init__(*args, **kwargs)

    def prepare_model(self, model, *args, **kwargs):
        return model

    def prepare(self, *args, **kwargs):
        prepped = []
        for obj in args:
            if isinstance(obj, nn.Module):
                prepped.append(self.prepare_model(obj, **kwargs))
            else:
                prepped.append(super().prepare(obj, **kwargs))
        return tuple(prepped) if len(prepped) > 1 else prepped[0]

class JOYTrainer(Trainer):

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        STRICT RESUME for adapter-only checkpoint:
        - do NOT load model weights (no model.safetensors(.index.json) required)
        - ONLY restore trainer_state / optimizer / scheduler / rng
        """
        if resume_from_checkpoint is None:
            return

        ckpt_dir = resume_from_checkpoint
        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"resume_from_checkpoint is not a dir: {ckpt_dir}")

        # 1) trainer_state.json
        state_path = os.path.join(ckpt_dir, TRAINER_STATE_NAME)
        if os.path.isfile(state_path):
            self.state = TrainerState.load_from_json(state_path)
        else:
            # 没有 trainer_state.json 就谈不上严格续训
            raise ValueError(f"Missing {TRAINER_STATE_NAME} in {ckpt_dir}, cannot strict-resume.")

        # 2) 确保 optimizer / scheduler 已创建（Trainer 的原逻辑通常会先 create 再 load，这里做防御）
        if self.optimizer is None:
            self.create_optimizer()
        if self.lr_scheduler is None:
            # 这里必须用 Trainer 的 create_scheduler，需要 num_training_steps
            # train() 里会算好 self.state.max_steps；如果没有就用 args.max_steps
            num_training_steps = self.state.max_steps if getattr(self.state, "max_steps", None) else self.args.max_steps
            if num_training_steps is None or num_training_steps <= 0:
                # fallback：至少别崩
                num_training_steps = self.args.max_steps if self.args.max_steps > 0 else 1
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

        # 3) optimizer.pt / scheduler.pt
        opt_path = os.path.join(ckpt_dir, OPTIMIZER_NAME)      # 通常是 optimizer.pt
        sch_path = os.path.join(ckpt_dir, SCHEDULER_NAME)      # 通常是 scheduler.pt

        if os.path.isfile(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu")
            cur = [len(g["params"]) for g in self.optimizer.state_dict()["param_groups"]]
            sav = [len(g["params"]) for g in opt_state["param_groups"]]
            print(f"[RESUME] optimizer group sizes: current={cur}, saved={sav}")
            self.optimizer.load_state_dict(opt_state)
        else:
            raise ValueError(f"Missing {OPTIMIZER_NAME} in {ckpt_dir}, cannot strict-resume.")

        if os.path.isfile(sch_path):
            sch_state = torch.load(sch_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(sch_state)
        else:
            raise ValueError(f"Missing {SCHEDULER_NAME} in {ckpt_dir}, cannot strict-resume.")

        # 4) rng_state（PyTorch 2.6+ 需要显式 weights_only=False）
        rng_path = None
        for name in _RNG_CANDIDATES:
            p = os.path.join(ckpt_dir, name)
            if os.path.isfile(p):
                rng_path = p
                break

        if rng_path is not None:
            try:
                # 显式关闭 weights_only（这是关键）
                rng_state = torch.load(
                    rng_path,
                    map_location="cpu",
                    weights_only=False,   # <<< 关键修复
                )
            except TypeError:
                # 兼容 PyTorch < 2.6
                rng_state = torch.load(rng_path, map_location="cpu")

            # 使用 Trainer 自带的 RNG 恢复逻辑
            try:
                self._load_rng_state(resume_from_checkpoint)
            except Exception as e:
                print(f"[WARN] Failed to restore RNG state strictly: {e}")
        else:
            raise ValueError(f"Missing rng_state in {ckpt_dir}, cannot strict-resume.")

        # 注意：这里故意不加载 model 权重，避免 index.json 问题
        # 并且 state.global_step 已从 trainer_state.json 恢复，Trainer 会用它跳过已训练 steps

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1) 只保存 adapter 权重（按 trainable 过滤最稳）
        adapter_state = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # 保险：只收集明显属于 LoRA/adapter 的参数，避免误收 base
            if (".default." not in n) and ("lora" not in n) and ("vblora" not in n) and ("vector_bank" not in n):
                continue
            k = _to_adapter_ckpt_key(n)
            adapter_state[k] = p.detach().cpu()

        if len(adapter_state) == 0:
            raise ValueError("[SAVE] No trainable adapter params found; refusing to save empty adapter checkpoint.")

        safe_save_file(adapter_state, os.path.join(output_dir, ADAPTER_WEIGHTS_NAME), metadata={"format": "pt"})

        # 2) 保存 adapter_config.json
        # 优先从模型对象取（如果有 peft_config），否则从 resume 的 ckpt 拷贝
        saved_cfg = False
        if hasattr(self.model, "peft_config"):
            try:
                cfg = self.model.peft_config
                # 兼容 dict / 单对象
                if isinstance(cfg, dict):
                    cfg = cfg.get("default", None) or next(iter(cfg.values()))
                if cfg is not None and hasattr(cfg, "save_pretrained"):
                    cfg.save_pretrained(output_dir)  # 会生成 adapter_config.json
                    saved_cfg = True
                elif cfg is not None:
                    # 兜底：写 json
                    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__
                    with open(os.path.join(output_dir, ADAPTER_CONFIG_NAME), "w", encoding="utf-8") as f:
                        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
                    saved_cfg = True
            except Exception:
                saved_cfg = False

        if not saved_cfg:
            src = getattr(self.args, "resume_from_checkpoint", None)
            if src:
                src_cfg = os.path.join(src, ADAPTER_CONFIG_NAME)
                if os.path.isfile(src_cfg):
                    shutil.copy2(src_cfg, os.path.join(output_dir, ADAPTER_CONFIG_NAME))
                    saved_cfg = True

        if not saved_cfg:
            raise ValueError("[SAVE] Cannot produce adapter_config.json (no model.peft_config and no resume ckpt to copy).")

    def _move_model_to_device(self, model, device):
        print('[JOYTrainer] Due to the placement feature in JOYSFT, skip moving model to', device)
        return model

    def _wrap_model(self, model, training=True, dataloader=None):
        self.model_wrapped = model
        return model

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {}
        if is_accelerate_available('0.28.0') and self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs
        if 'num_steps' in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                raise ValueError("The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`.")
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs['num_steps']
        accelerator_config = self.args.accelerator_config.to_dict()
        if is_accelerate_available('0.28.0'):
            dataloader_params = ['split_batches', 'dispatch_batches', 'even_batches', 'use_seedable_sampler']
            dataloader_config_dict = {param: accelerator_config.pop(param) for param in dataloader_params if param in accelerator_config}
            if DataLoaderConfiguration is None:
                raise ImportError('Your accelerate does not provide DataLoaderConfiguration but Trainer expects it.')
            dataloader_config = DataLoaderConfiguration(**dataloader_config_dict)
            if is_accelerate_available('1.1.0'):
                dataloader_config.data_seed = self.args.data_seed
        else:
            dataloader_config = None
        non_blocking = accelerator_config.pop('non_blocking', False)
        if not is_accelerate_available('0.30.0'):
            if non_blocking:
                raise ImportError('`non_blocking` is only supported in accelerate v0.30.0 and above. Please upgrade accelerate to use this feature.')
        else:
            if non_blocking and (not self.args.dataloader_pin_memory):
                logger.warning('`non_blocking` is enabled but `dataloader_pin_memory` is not. For best performance, enable both.')
            if dataloader_config is not None:
                dataloader_config.non_blocking = non_blocking
        accelerator_config.pop('gradient_accumulation_kwargs', None)
        args = {'deepspeed_plugin': self.args.deepspeed_plugin, 'device_placement': False}
        if is_accelerate_available('0.28.0'):
            args['dataloader_config'] = dataloader_config
        else:
            args.update(accelerator_config)
        if getattr(self.args, 'tp_size', 1) > 1:
            self.is_tp_enabled = True
            if version.parse(accelerate_version) > version.parse('1.3.0') and TorchTensorParallelPlugin is not None:
                args['torch_tp_plugin'] = TorchTensorParallelPlugin(tp_size=self.args.tp_size)
            else:
                raise ValueError('Requires accelerate>1.3.0 to use Tensor Parallelism.')
        self.accelerator = JOYAccelerator(**args)
        try:
            self.accelerator.state.device_ids = [0]
            self.accelerator.state.num_processes = 1
            self.accelerator.state.num_gpus = 1
        except Exception:
            pass
        self.gather_function = self.accelerator.gather_for_metrics
        if 'use_gather_object' in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(self.gather_function, use_gather_object=self.args.eval_use_gather_object)
        self.is_deepspeed_enabled = getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, 'fsdp_plugin', None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, 'torch_tp_plugin', None) is not None
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ['limit_all_gathers', 'activation_checkpointing']:
                setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic when using FSDP.")
        if self.is_deepspeed_enabled and getattr(self.args, 'hf_deepspeed_config', None) is None:
            self.propagate_args_to_deepspeed()
        if self.args.save_only_model and (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.load_best_model_at_end:
            wrapper = 'DeepSpeed' if self.is_deepspeed_enabled else 'FSDP'
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")
        if self.is_deepspeed_enabled and self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.args.auto_find_batch_size:
            raise ValueError("`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP")
        if self.args.save_only_model and self.is_fsdp_enabled and ('SHARDED_STATE_DICT' in str(self.accelerator.state.fsdp_plugin.state_dict_type)):
            raise ValueError("save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'")
        if dataloader_config is not None:
            dataloader_config.split_batches = False
            dataloader_config.dispatch_batches = False
            dataloader_config.even_batches = False

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with per_device_train_batch_size
        (no implicit multipliers by number of visible GPUs).
        """
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available():
            try:
                import datasets
                if isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description='training')
                else:
                    data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
            except Exception:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
        dataloader_params = {'batch_size': self.args.per_device_train_batch_size, 'collate_fn': data_collator, 'num_workers': self.args.dataloader_num_workers, 'pin_memory': self.args.dataloader_pin_memory, 'persistent_workers': self.args.dataloader_persistent_workers}
        dataloader_params["num_workers"] = 0
        dataloader_params["pin_memory"] = False
        dataloader_params["persistent_workers"] = False
        if not isinstance(train_dataset, IterableDataset):
            dataloader_params['sampler'] = self._get_train_sampler()
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker
            if self.args.dataloader_num_workers > 0 and self.args.dataloader_prefetch_factor is not None:
                dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor
        dl = DataLoader(train_dataset, **dataloader_params)
        print(
            f"[DL CFG] num_workers={dl.num_workers}, "
            f"pin_memory={dl.pin_memory}, "
            f"persistent_workers={getattr(dl, 'persistent_workers', None)}, "
            f"prefetch_factor={getattr(dl, 'prefetch_factor', None)}"
        )
        prepared = self.accelerator.prepare(dl, device_placement=[False])  # 或 except fallback
        try:
            print(
                f"[PREP DL CFG] num_workers={prepared.num_workers}, "
                f"pin_memory={prepared.pin_memory}, "
                f"persistent_workers={getattr(prepared, 'persistent_workers', None)}"
            )
        except TypeError:
            prepared = self.accelerator.prepare(dl)
        return prepared

    def training_step(self, model: torch.nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        step_start = time.time()
        model.train()
        if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        del inputs
        step_cost = time.time() - step_start
        dbg_every = 10
        if (self.state.global_step % dbg_every) == 0:
            print(
                f"[JOY-STEP] global_step={self.state.global_step} "
                f"training_step_cost={step_cost:.2f}s "
                f"loss={loss.detach().float().item():.4f}"
            )
        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version='2.0'):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning('`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache().')
            else:
                torch.cuda.empty_cache()
        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs['learning_rate'] = self._get_learning_rate()
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            if getattr(self.accelerator, 'distributed_type', None) and str(self.accelerator.distributed_type) == 'DistributedType.DEEPSPEED':
                kwargs['scale_wrt_gas'] = False
            self.accelerator.backward(loss, **kwargs)
        ret = loss.detach()
        if ret.device != self.args.device:
            ret = ret.to(self.args.device, non_blocking=True)
        
        # [内存泄露侦探] 增强版
        # ----------------------------------------------------------------------
        if self.state.global_step > 0 and self.state.global_step % 20 == 0:
            import warnings
            
            # 1. 强制刷新，告诉终端我要开始输出了
            print(f"\n======== [Memory Hunter] Step {self.state.global_step} Analysis ========", flush=True)
            
            # 2. 强制 GC
            gc.collect()
            
            # 3. 统计 Tensor (增加警告屏蔽，防止 DeepSpeed/KT 对象报错刷屏)
            # 使用 catch_warnings 忽略那个 annoying 的 FutureWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") 
                try:
                    # 获取所有对象
                    all_objects = gc.get_objects()
                    
                    # 过滤 Tensor，加个 try-except 防止某些 weird 对象在 isinstance 时崩溃
                    tensors = []
                    for obj in all_objects:
                        try:
                            if isinstance(obj, torch.Tensor):
                                tensors.append(obj)
                        except:
                            # 某些 C++ 扩展对象在被访问时可能会报错，直接跳过
                            pass
                            
                except Exception as e:
                    print(f"Error during object scan: {e}", flush=True)
                    tensors = []

            gpu_tensors = [t for t in tensors if t.is_cuda]
            cpu_tensors = [t for t in tensors if not t.is_cuda]
            
            # 必须加 flush=True，否则在训练 log 中看不到
            print(f" >> [Tensor Stats] Total: {len(tensors)} | GPU: {len(gpu_tensors)} | CPU: {len(cpu_tensors)}", flush=True)
            
            # 打印 Top 3 大的 CPU Tensor
            large_cpu_tensors = sorted(cpu_tensors, key=lambda t: t.nelement() * t.element_size(), reverse=True)[:3]
            if large_cpu_tensors:
                print("    Top 3 Large CPU Tensors:", flush=True)
                for t in large_cpu_tensors:
                    print(f"      Shape: {t.shape}, Dtype: {t.dtype}, Size: {t.nelement() * t.element_size() / 1024 / 1024:.2f} MB", flush=True)

            # 3. 对象类型计数 (Diff)
            current_counts = collections.Counter()
            for obj in all_objects: # 复用上面的 list，省时间
                t_name = type(obj).__name__
                current_counts[t_name] += 1
            
            last_counts = getattr(self, "_last_mem_counts", None)
            
            if last_counts:
                print(" >> [Object Growth] Types increasing in count:", flush=True)
                growth_found = False
                diff = current_counts - last_counts
                for t_name, count in diff.most_common(10):
                    if count > 0:
                        growth_found = True
                        print(f"    Type '{t_name}': +{count} (Total: {current_counts[t_name]})", flush=True)
                
                if not growth_found:
                    print("    (No python object count leak.)", flush=True)
            
            self._last_mem_counts = current_counts
            
            # 清理
            del all_objects, tensors, gpu_tensors, cpu_tensors, current_counts, last_counts
            gc.collect()
            print("==============================================================\n", flush=True)

        if self.state.global_step % 10 == 0:
            flush_memory()

        return ret

class SFTJsonListDataset(TorchDataset):

    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int=512):
        super().__init__()
        with open(path, 'r', encoding='utf-8') as f:
            self.samples: List[Dict] = json.load(f)
        self.tok = tokenizer
        self.max_len = max_len

    @staticmethod
    def build_example(ins: str, inp: str, out: str) -> Dict[str, str]:
        ins = (ins or '').strip()
        inp = (inp or '').strip()
        out = (out or '').strip()
        prompt = ins + inp if ins else inp
        return {'prompt': prompt, 'response': out}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        eg = self.build_example(rec.get('instruction', ''), rec.get('input', ''), rec.get('output', ''))
        prompt_ids = self.tok(eg['prompt'], max_length=self.max_len, truncation=True, add_special_tokens=False)['input_ids']
        response_ids = self.tok(eg['response'], max_length=self.max_len, truncation=True, add_special_tokens=False)['input_ids']
        eos_id = self.tok.eos_token_id
        input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])
        input_ids = input_ids[:self.max_len]
        labels = [-100] * min(len(prompt_ids), self.max_len)
        tail = input_ids[len(labels):]
        labels = labels + tail
        labels = labels[:self.max_len]
        attention_mask = [1] * len(input_ids)
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), 'labels': torch.tensor(labels, dtype=torch.long), 'attention_mask': torch.tensor(attention_mask, dtype=torch.long)}

def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path):
    Path(save_adapter_path).mkdir(parents=True, exist_ok=True)
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'shared_experts.gate_proj', 'shared_experts.up_proj', 'shared_experts.down_proj'], r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    train_dataset = SFTJsonListDataset(sft_data_path, tokenizer, max_len=512)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(output_dir=save_adapter_path, per_device_train_batch_size=1, gradient_accumulation_steps=16, num_train_epochs=1, learning_rate=0.0001, fp16=False, logging_steps=10, save_steps=200, dataloader_drop_last=True, ddp_find_unused_parameters=False)
    debug_path = os.path.join(save_adapter_path, 'model_infra_debug.json')
    with open(debug_path, 'w', encoding='utf-8') as f:
        json.dump({'model': str(model)}, f, ensure_ascii=False, indent=2)
    trainer = JOYTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, data_collator=data_collator)
    model.config.use_cache = False
    trainer.train()

def inject_lora_layer(model, use_adapter_path):
    cfg_path = os.path.join(use_adapter_path, 'adapter_config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    task_type_str = (data.get('task_type') or 'CAUSAL_LM').upper()
    bias = data.get('bias', 'none')
    if bias in (None, False):
        bias = 'none'
    if data.get('lora_bias') is True and bias == 'none':
        bias = 'lora_only'
    tmods = data.get('target_modules')
    if isinstance(tmods, str):
        tmods = [m.strip() for m in tmods.split(',') if m.strip()]
    mts = data.get('modules_to_save', None)
    if isinstance(mts, str):
        mts = [m.strip() for m in mts.split(',') if m.strip()]
    rank_pattern = data.get('rank_pattern') or None
    alpha_pattern = data.get('alpha_pattern') or None
    lora_config = LoraConfig(r=data.get('r', 8), lora_alpha=data.get('lora_alpha', 32), lora_dropout=float(data.get('lora_dropout', 0.0)), bias=bias, task_type=TaskType[task_type_str], target_modules=tmods, modules_to_save=mts, init_lora_weights=bool(data.get('init_lora_weights', True)), inference_mode=bool(data.get('inference_mode', True)), use_rslora=bool(data.get('use_rslora', False)), use_dora=bool(data.get('use_dora', False)))
    print(f'lora_config:{lora_config.__dict__}')
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.eval()