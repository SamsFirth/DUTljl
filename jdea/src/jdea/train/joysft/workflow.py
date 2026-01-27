from typing import TYPE_CHECKING, Optional
from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
logger = get_logger(__name__)
from ...extras.constants import EngineName
import os

def _resolve_adapter_dir(resume_ckpt: str) -> str:
    # 常见候选：ckpt 本身、ckpt 的父目录（有的人把 adapter 存在 output_dir 根）
    candidates = [resume_ckpt, os.path.dirname(resume_ckpt)]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        has_cfg = os.path.isfile(os.path.join(d, "adapter_config.json"))
        has_st = os.path.isfile(os.path.join(d, "adapter_model.safetensors"))
        has_gguf = any(fn.endswith(".gguf") for fn in os.listdir(d))
        if has_cfg and (has_st or has_gguf):
            return d
    # 找不到就先返回 ckpt，让后续报错更明确
    return resume_ckpt

def run_sft(model_args: 'ModelArguments', data_args: 'DataArguments', training_args: 'Seq2SeqTrainingArguments', finetuning_args: 'FinetuningArguments', generating_args: 'GeneratingArguments', callbacks: Optional[list['TrainerCallback']]=None):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage='sft', **tokenizer_module)

    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt:
        adapter_dir = _resolve_adapter_dir(resume_ckpt)

        model_args.adapter_name_or_path = [adapter_dir]   # ✅ adapter 从这里加载
        if model_args.use_joysft:
            model_args.infer_backend = EngineName.JOYSFT

        finetuning_args.create_new_adapter = False        # ✅ 严格续训必须禁用新建

        print("[RESUME] resume_ckpt =", resume_ckpt)
        print("[RESUME] adapter_dir =", adapter_dir)

    def _count_trainable(model):
        return sum(1 for _n, p in model.named_parameters() if p.requires_grad)

    # 现在再 load_model：内部 init_adapter 会负责把 adapter 注入并加载
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    print("[SANITY] trainable tensors =", _count_trainable(model))

    # 如果是 0，直接把参数名里看起来像 adapter/lora 的打开（先救火）
    if training_args.do_train and resume_ckpt and _count_trainable(model) == 0:
        for _n, p in model.named_parameters():
            p.requires_grad_(False)

        hit = 0
        for n, p in model.named_parameters():
            if ("lora" in n) or (".default." in n) or ("vblora" in n) or ("vector_bank" in n):
                p.requires_grad_(True)
                hit += 1

        print("[SANITY] force-enabled adapter params =", hit)
        print("[SANITY] trainable tensors(after) =", _count_trainable(model))

    from joyaiSFT.util.globals import GLOBAL_CONFIG
    GLOBAL_CONFIG._config['mod'] = 'sft'
    if getattr(model, 'is_quantized', False) and (not training_args.do_train):
        setattr(model, '_hf_peft_config_loaded', True)
    data_collator = SFTDataCollatorWith4DAttentionMask(template=template, model=model if not training_args.predict_with_generate else None, pad_to_multiple_of=8 if training_args.do_train else None, label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id, block_diag_attn=model_args.block_diag_attn, attn_implementation=getattr(model.config, '_attn_implementation', None), compute_dtype=model_args.compute_dtype, **tokenizer_module)
    metric_module = {}
    if training_args.predict_with_generate:
        raise NotImplementedError('`predict_with_generate` is not supported in JOYSFT SFT yet.')
    elif finetuning_args.compute_accuracy:
        raise NotImplementedError('`compute_accuracy` is not supported in JOYSFT SFT yet.')
    from joyaiSFT.sft.lora import JOYTrainer
    trainer = JOYTrainer(model=model, args=training_args, tokenizer=tokenizer_module, data_collator=data_collator, callbacks=callbacks, **dataset_module, **metric_module)
    trainer.model_accepts_loss_kwargs = False
    if training_args.do_train:
        model.config.use_cache = False
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics['effective_tokens_per_sec'] = calculate_tps(dataset_module['train_dataset'], train_result.metrics, stage='sft')
        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ['loss']
            if isinstance(dataset_module.get('eval_dataset'), dict):
                keys += sum([[f'eval_{key}_loss', f'eval_{key}_accuracy'] for key in dataset_module['eval_dataset'].keys()], [])
            else:
                keys += ['eval_loss', 'eval_accuracy']
            plot_loss(training_args.output_dir, keys=keys)
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)