import argparse
import glob
import os
import torch
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', help='Path to LLaVA v1.5 model')
args = ap.parse_args()
path = sorted(glob.glob(f'{args.model}/pytorch_model*.bin'))[-1]
checkpoint = torch.load(path)
mm_tensors = [k for k, v in checkpoint.items() if k.startswith('model.mm_projector')]
projector = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(projector, f'{args.model}/llava.projector')
clip_tensors = [k for k, v in checkpoint.items() if k.startswith('model.vision_tower')]
if len(clip_tensors) > 0:
    clip = {name.replace('vision_tower.vision_tower.', ''): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f'{args.model}/llava.clip')
    if os.path.exists(f'{args.model}/added_tokens.json'):
        with open(f'{args.model}/added_tokens.json', 'w') as f:
            f.write('{}\n')
print('Done!')
print(f'Now you can convert {args.model} to a regular LLaMA GGUF file.')
print(f'Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.')