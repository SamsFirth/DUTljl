from torch.profiler import profile, record_function, ProfilerActivity
import os
from transformers import TrainerCallback

class ProfilerCallback(TrainerCallback):

    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

def _short(t):
    return tuple(t.shape) if isinstance(t, torch.Tensor) else type(t)

def install_shape_probes(model):
    if os.environ.get('KT_DEBUG_MOE', '0') != '1':
        print('[KT_DEBUG_MOE] off')
        return
    try:
        acc = trainer.accelerator
        cfg = getattr(acc, 'dataloader_config', None)
        if cfg is not None:
            print('[ACCEL DL CONFIG]', 'split_batches=', getattr(cfg, 'split_batches', None), 'dispatch_batches=', getattr(cfg, 'dispatch_batches', None), 'even_batches=', getattr(cfg, 'even_batches', None), 'use_seedable_sampler=', getattr(cfg, 'use_seedable_sampler', None), 'non_blocking=', getattr(cfg, 'non_blocking', None))
    except Exception as e:
        print('[ACCEL DL CONFIG] <err>', e)
    try:
        emb = model.base_model.model.model.embed_tokens

        def _emb_pre(mod, inp):
            x = inp[0]
            if not hasattr(mod, '_dbg_once'):
                print(f'[DBG] embed input_ids shape = {tuple(x.shape)}  (expect B,S)')
                mod._dbg_once = True
        emb.register_forward_pre_hook(_emb_pre)
    except Exception as e:
        print('[DBG] embed hook failed:', e)
    try:
        first_layer = model.base_model.model.model.layers[0]
        _orig_fwd = first_layer.forward

        def _wrap_fwd(self, *args, **kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if not hasattr(self, '_dbg_once_in'):
                print(f'[DBG] L0.in hidden_states = {_short(hs)}  (expect B,S,H)')
                self._dbg_once_in = True
            out = _orig_fwd(*args, **kwargs)
            hs_out = out[0] if isinstance(out, (tuple, list)) else out
            if not hasattr(self, '_dbg_once_out'):
                print(f'[DBG] L0.out hidden_states = {_short(hs_out)}')
                self._dbg_once_out = True
            return out
        first_layer.forward = MethodType(_wrap_fwd, first_layer)
    except Exception as e:
        print('[DBG] L0 wrap failed:', e)
    try:
        moe_layer = None
        for i, lyr in enumerate(model.base_model.model.model.layers):
            if hasattr(lyr, 'mlp'):
                moe_layer = lyr.mlp
                moe_idx = i
                break
        if moe_layer is not None:
            _moe_orig = moe_layer.forward

            def _moe_wrap(self, *args, **kwargs):
                x = args[0] if args else kwargs.get('hidden_states')
                if not hasattr(self, '_dbg_once'):
                    print(f'[DBG] MLP(in) @layer{moe_idx} hidden_states = {_short(x)}')
                    if isinstance(x, torch.Tensor) and x.dim() == 3:
                        B, S, H = x.shape
                        print(f'[DBG] tokens before flatten = B*S = {B}*{S} = {B * S}')
                    self._dbg_once = True
                return _moe_orig(*args, **kwargs)
            moe_layer.forward = MethodType(_moe_wrap, moe_layer)
        else:
            print('[DBG] no moe_layer found')
    except Exception as e:
        print('[DBG] moe wrap failed:', e)
    try:
        from joyaiSFT.operators.experts import JOYSFTExperts

        def _experts_pre(mod, args):
            if hasattr(mod, '_dbg_once'):
                return
            try:
                input_tensor, expert_ids, weights = args[:3]
                print(f'[DBG] experts.in input_tensor={tuple(input_tensor.shape)} expert_ids={tuple(expert_ids.shape)} weights={tuple(weights.shape)}')
                if input_tensor.dim() == 2:
                    N = input_tensor.shape[0]
                    print(f'[DBG] N(input rows)={N}')
                if expert_ids.dim() == 2:
                    T, K = expert_ids.shape
                    print(f'[DBG] tokens(T)={T}, K={K}, T*K={T * K}')
                mod._dbg_once = True
            except Exception as e:
                print('[DBG] experts hook parse err:', e)
        count = 0
        for name, m in model.named_modules():
            if isinstance(m, JOYSFTExperts):
                m.register_forward_pre_hook(_experts_pre)
                count += 1
        print(f'[KT_DEBUG_MOE] installed experts hook on {count} modules.')
    except Exception as e:
        print('[DBG] experts hook failed:', e)

def inspect_device(model, write_file):
    for name, module in model.named_modules():
        with open(write_file, 'a') as file:
            file.write(f'Layer: {name}\n')
        for param_name, param in module.named_parameters(recurse=False):
            with open(write_file, 'a') as file:
                file.write(f"  Parameter '{param_name}' device: {param.device}\n")
        for buffer_name, buffer in module.named_buffers(recurse=False):
            with open(write_file, 'a') as file:
                file.write(f"  Buffer '{buffer_name}' device: {buffer.device}\n")

def print_model_params(model):
    for layer_idx in range(0, 3):
        layer = model.model.orig_module.layers[layer_idx]
        print(f'\n================ Layer {layer_idx} Attention ================')
        q_proj = layer.self_attn.orig_module.q_proj.orig_module
        print(f'\nq_proj.generate_linear.weight (shape: {q_proj.generate_linear.weight.shape})')
        print(q_proj.generate_linear.weight.cpu())

def print_lora_params(model):
    for layer_idx in range(0, 3):
        layer = model.base_model.model.model.orig_module.layers[layer_idx]
        q_proj_module = layer.self_attn.orig_module.q_proj.orig_module
        linear_weight = q_proj_module.generate_linear.weight
        lora_A_weight = q_proj_module.lora_A['default'].weight
        lora_B_weight = q_proj_module.lora_B['default'].weight
        print(f'\n=================== Layer {layer_idx} ===================')
        print('\nOriginal Linear (first row slice):')
        print(linear_weight.cpu())
        print('\nLora_A (first row slice):')
        print(lora_A_weight.cpu())
        print('\nLora_B (first row slice):')
        print(lora_B_weight.cpu())

def print_grad_fn(grad_fn, indent=0):
    """递归打印计算图节点"""
    if grad_fn is None:
        return
    print(' ' * indent, f"Node: {str(grad_fn).split('(')[0]}")
    print(' ' * indent, f'  Metadata: {grad_fn.metadata}')
    for child in getattr(grad_fn, 'next_functions', []):
        if child[0] is not None:
            print_grad_fn(child[0], indent + 2)

def forward_hook(module, inputs, output):
    if isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            if o is None:
                print(f'{module.__class__.__name__} output index {i} is None')
            else:
                print(f'{module.__class__.__name__} output index {i}: requires_grad={o.requires_grad}, grad_fn={o.grad_fn}')
    elif output is None:
        print(f'{module.__class__.__name__} returned None')
    else:
        print(f'{module.__class__.__name__}: requires_grad={output.requires_grad}, grad_fn={output.grad_fn}')

def check_moe_gradients(model):
    moe_layer = model.base_model.model.model.orig_module.layers[1].mlp.orig_module
    for name, param in moe_layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad)
            print(f'MoE参数 {name} 梯度范数: {grad_norm}')
        else:
            print(f'MoE参数 {name} 无梯度')

def disable_all_dropout(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            child.p = 0
            child.inplace = False
        disable_all_dropout(child)

def verify_lora_layers(model):
    for layer_path in target_layers:
        module = model.get_submodule(layer_path)
        orig_module = module.orig_module
        W = orig_module.weight.data
        lora_A = module.lora_A['default'].weight.data
        lora_B = module.lora_B['default'].weight.data
        alpha_over_r = 32 / 8
        input_tensor = layer_data[layer_path]['input']
        try:
            original_output = torch.matmul(input_tensor, W)
        except:
            original_output = torch.matmul(input_tensor, W.T)
        lora_effect = torch.matmul(torch.matmul(input_tensor, lora_A.T), lora_B.T) * alpha_over_r
        manual_output = original_output + lora_effect
        model_output = layer_data[layer_path]['output']
        print(f'manual_output:{manual_output}')
        print(f'model_output:{model_output}')
        if torch.allclose(manual_output, model_output, atol=1e-05):
            print(f'{layer_path} 验证通过')
        else:
            print(f'{layer_path} 验证失败！最大误差：{torch.max(torch.abs(manual_output - model_output))}')

def print_moe_stats(moe_layer: JOYExpertsTorch):
    print(f'Total Params: {moe_layer.total_params / 1000000.0:.2f}M')
    total_time = sum(moe_layer.times)
    gflops = moe_layer.total_flops / 1000000000.0 / total_time if total_time != 0 else 0
    print(f'Total Calls: {moe_layer.call_count}')
    print(f'Overall GFLOPS: {gflops:.2f}')
    if moe_layer.call_count > 0:
        last_flops = moe_layer.flops_per_call[-1]
        last_time = moe_layer.times[-1]
        print(f'\nLast Call - FLOPs: {last_flops / 1000000000.0:.2f}G  Time: {last_time * 1000:.2f}ms  GFLOPS: {last_flops / 1000000000.0 / last_time:.2f}')

def recursive_traverse(model, parent_name=''):
    """
    递归遍历模型，查找MoE层并调用print_moe_stats。
    """
    for name, module in model.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name
        if isinstance(module, JOYSFTExperts):
            print(f'Found MoE layer: {full_name}')
            print_moe_stats(module.generate_experts)
        recursive_traverse(module, full_name)

def log_step_state(step: int, inputs: dict, loss: torch.Tensor, model: nn.Module, log_dir: str='train_logs'):
    """
    把当前 step 的输入 / loss / grad / param 保存到 log_dir/step_{step}.pt
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logged_inputs = {k: v.detach().cpu() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    loss_val = loss.detach().cpu()
    params, grads = ({}, {})
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu()
        grads[name] = p.grad.detach().cpu() if p.grad is not None else None
    torch.save({'step': step, 'inputs': logged_inputs, 'loss': loss_val, 'params': params, 'grads': grads}, f'{log_dir}/step_{step:08d}.pt')

def collect_gradients(model, input_ids):
    torch.manual_seed(42)
    output = model(input_ids=input_ids)
    logits = output.logits
    loss = logits.mean()
    model.zero_grad()
    loss.backward()
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(f'{name}: {param.grad.norm().item():.6f}')
    return grads

def report_meta_tensors(model):
    import torch, inspect
    meta_modules = []
    for mod_name, mod in model.named_modules():
        metas = []
        for n, p in list(mod.named_parameters(recurse=False)):
            if getattr(p, 'is_meta', False) and p.is_meta:
                metas.append(('param', n, tuple(p.shape)))
        for n, b in list(mod.named_buffers(recurse=False)):
            if getattr(b, 'is_meta', False) and b.is_meta:
                metas.append(('buffer', n, tuple(b.shape)))
        if metas:
            print(f'[META] {mod_name} ({type(mod).__name__}): {metas}')
            meta_modules.append((mod_name, type(mod).__name__, metas))
    return meta_modules
    '\n    # multi-gpu dataloader test\n    # _ = report_meta_tensors(model)\n    \n    # print("=== SAMPLE INSPECT ===")\n    # for i in range(2):\n    #     summary = {}\n    #     for k,v in ex.items():\n    #         if isinstance(v, list):\n    #             if len(v)>0 and isinstance(v[0], list):\n    #                 summary[k] = f"list-of-lists len={len(v)} x len0={len(v[0])}"\n    #             else:\n    #                 summary[k] = f"list len={len(v)}"\n    #         elif torch.is_tensor(v):\n    #             summary[k] = f"tensor shape={tuple(v.shape)}"\n    #         else:\n    #             summary[k] = str(type(v))\n    #     print(f"[SAMPLE {i}]", summary)\n    \n    # trainer.accelerator = Accelerator(device_placement=False)\n    # first_batch = next(iter(trainer.get_train_dataloader()))\n    # print("Batch keys:", list(first_batch.keys()))\n    \n    # acc = JOYAccelerator(device_placement=False)\n    # acc.state.device_ids = [0]\n    # acc.state.num_processes = 1\n    # acc.state.num_gpus = 1\n    # trainer.accelerator = acc\n\n    # print("Accelerator device_ids:", trainer.accelerator.state.device_ids)\n    # print(f"type(trainer.model):{type(trainer.model)}")\n    # print(f"type(trainer.accelerator):{type(trainer.accelerator)}")\n    \n    \n    # print("-------------------------START TRAINING!!!-------------------------")\n\n    # cfg = getattr(trainer.accelerator, "dataloader_config", None)\n    # print(\n    #     "[ACCEL DL CONFIG]",\n    #     "split_batches=", getattr(cfg, "split_batches", None),\n    #     "dispatch_batches=", getattr(cfg, "dispatch_batches", None),\n    #     "even_batches=", getattr(cfg, "even_batches", None),\n    #     "use_seedable_sampler=", getattr(cfg, "use_seedable_sampler", None),\n    #     "non_blocking=", getattr(cfg, "non_blocking", None),\n    # )\n    # print("--------------------NEW DEBUG--------------------")\n    # install_shape_probes(trainer.model) # print some debug info about multi-gpu placement.\n\n    # input_ids = torch.randint(0, 1000, (32, 128), device="cuda:0")\n    # gradients = collect_gradients(model, input_ids)\n    '
    '\n    ----------------------- START: Lora Test -----------------------\n    \n\n    # for name, module in model.named_modules():\n    #     if "q_proj" in name or "kv_a_proj" in name or "o_proj" in name:\n    #         print(name)\n\n    # print_model_params(model)\n\n    # model = JOYSFTLinearLora()\n\n    # inspect_device(model, \'/home/yj/joyaiSFT/device1.txt\')\n    # with open(\'/home/yj/joyaiSFT/device1.txt\', \'a\') as file:\n    #     file.write(f"Base model device: {model.base_model.device}\n")\n        # file.write(f"LoRA adapter device: {model.lora_config[\'target_modules\'].device}\n")\n    # print(f"Base model device: {model.base_model.device}") \n    # print(f"LoRA adapter device: {model.lora_config[\'target_modules\'].device}") \n\n\n    # model = model.to(\'cuda\')\n\n    # for name, module in model.named_modules():\n    #     module.register_forward_hook(forward_hook)\n\n    # for name, parms in model.named_parameters():\t\n    #     # parms.requires_grad = True\n    #     print(\'-->name:\', name)\n    #     print(\'-->para:\', parms)\n    #     print(\'-->grad_requirs:\',parms.requires_grad)\n    #     print(\'-->grad_fn:\',parms.grad_fn)\n    #     print(\'-->grad_value:\',parms.grad)\n    #     print("===")\n\n    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))\n    # loss = output.logits.mean()\n\n    # dot = make_dot(loss, params=dict(model.named_parameters()))\n    # dot.render("KT_compute_graph", format="svg")\n\n    # inspect_device(model, \'/home/yj/joyaiSFT/device2.txt\')\n    # with open(\'/home/yj/joyaiSFT/device2.txt\', \'a\') as file:\n    #     file.write(f"Base model device: {model.base_model.device}\n")\n        # file.write(f"LoRA adapter device: {model.lora_config[\'target_modules\'].device}\n")\n    # print(f"Base model device: {model.base_model.device}") \n    # print(f"LoRA adapter device: {model.lora_config[\'target_modules\'].device}") \n\n    # print_lora_params(model)\n\n    # trainer = JOYTrainer(\n    #     model=model,\n    #     train_dataset=train_dataset,\n    #     args=transformers.TrainingArguments(\n    #         output_dir=save_adapter_path,\n    #         per_device_train_batch_size=1,\n    #         gradient_accumulation_steps=16,\n    #         num_train_epochs=10,\n    #         learning_rate=3e-4,\n    #         fp16=False,\n    #         logging_steps=10,\n    #         save_steps=200,\n    #         dataloader_drop_last=True,\n    #         ddp_find_unused_parameters=False \n    #     ),\n    #     data_collator=DataCollatorForSeq2Seq(\n    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True\n    #     ),\n    # )\n\n    # model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))\n\n    # trainer.train()\n\n    # print_lora_params(model)\n\n    # model = model.merge_and_unload()\n    ----------------------- END: Lora Test -----------------------\n\n    '