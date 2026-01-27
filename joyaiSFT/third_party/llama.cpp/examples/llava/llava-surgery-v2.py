import argparse
import glob
import os
import torch
from safetensors.torch import load as safe_load, save as safe_save, safe_open, save_file

def is_safetensor_file(file_path):
    return file_path.endswith('.safetensors')

def load_model(file_path):
    if is_safetensor_file(file_path):
        tensors = {}
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).clone()
                print(f'{key} : {tensors[key].shape}')
        return (tensors, 'safetensor')
    else:
        return (torch.load(file_path, map_location=torch.device('cpu')), 'pytorch')

def save_model(model, file_path, file_type):
    if file_type == 'safetensor':
        save_file(model, file_path)
    else:
        torch.save(model, file_path)

def clean_vision_tower_from_checkpoint(checkpoint_path):
    checkpoint, file_type = load_model(checkpoint_path)
    model_path = os.path.dirname(checkpoint_path)
    print(f'Searching for vision tower tensors in {checkpoint_path}')
    clip_tensors = [k for k, v in checkpoint.items() if k.startswith('model.vision_tower') or k.startswith('vit.')]
    if len(clip_tensors) > 0:
        print(f'Found {len(clip_tensors)} tensors to extract from {checkpoint_path}')
        clip_path = os.path.join(model_path, 'llava.clip')
        if os.path.exists(clip_path):
            print(f'Loading existing llava.clip from {clip_path}')
            existing_clip, _ = load_model(clip_path)
        else:
            print(f'Creating new llava.clip at {clip_path}')
            existing_clip = {}
        for name in clip_tensors:
            simple_name = name[name.index('vision_model.'):] if 'vision_model.' in name else name
            print(f'Adding {simple_name} to llava.clip')
            if simple_name not in existing_clip:
                existing_clip[simple_name] = checkpoint[name]
        save_model(existing_clip, clip_path, 'pytorch')
        for name in clip_tensors:
            del checkpoint[name]
        checkpoint_path = checkpoint_path
        return True
    return False

def find_relevant_checkpoints(checkpoint_paths, newline_criteria, projector):
    newline_checkpoint_path = None
    projector_checkpoint_path = None
    for path in checkpoint_paths:
        checkpoint, _ = load_model(path)
        if newline_criteria(checkpoint) and newline_checkpoint_path is None:
            newline_checkpoint_path = path
        if projector(checkpoint):
            projector_checkpoint_path = path
    return (newline_checkpoint_path, projector_checkpoint_path)

def newline_criteria(checkpoint):
    return any((k.startswith('model.image_newline') for k in checkpoint.keys()))

def proj_criteria(checkpoint):
    return any((k.startswith('model.mm_projector') or k.startswith('vision_proj.') for k in checkpoint.keys()))
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to LLaVA v1.5+ model')
ap.add_argument('-C', '--clean-vision-tower', action='store_true', help='Remove any vision tower from the model files')
args = ap.parse_args()
if args.clean_vision_tower:
    model_files = sorted(glob.glob(f'{args.model}/*'), key=os.path.getmtime, reverse=True)
    checkpoint_paths = [path for path in model_files if path.endswith('.bin') and 'pytorch' in path.split('/')[-1].split('\\')[-1] or (path.endswith('.safetensors') and 'model' in path.split('/')[-1].split('\\')[-1])]
    for projector_checkpoint_path in checkpoint_paths:
        print(f'Cleaning {projector_checkpoint_path}')
        if not clean_vision_tower_from_checkpoint(projector_checkpoint_path):
            print(f'No vision tower found in {projector_checkpoint_path}')
    print('Done! All vision tower tensors are removed from the model files and stored in llava.clip file.')
model_files = sorted(glob.glob(f'{args.model}/*'), key=os.path.getmtime, reverse=True)
checkpoint_paths = [path for path in model_files if path.endswith('.bin') and 'pytorch' in path.split('/')[-1].split('\\')[-1] or (path.endswith('.safetensors') and 'model' in path.split('/')[-1].split('\\')[-1])]
newline_checkpoint_path, projector_checkpoint_path = find_relevant_checkpoints(checkpoint_paths, newline_criteria, proj_criteria)
print(f'Taking projector from {projector_checkpoint_path}')
first_mm_tensors = []
first_checkpoint = None
if newline_checkpoint_path is not None:
    print(f'Taking newline from {newline_checkpoint_path}')
    first_checkpoint, file_type = load_model(newline_checkpoint_path)
    first_mm_tensors = [k for k, v in first_checkpoint.items() if k.startswith('model.image_newline')]
mm_tensors = []
last_checkpoint = None
if projector_checkpoint_path is not None:
    last_checkpoint, file_type = load_model(projector_checkpoint_path)
    mm_tensors = [k for k, v in last_checkpoint.items() if k.startswith('model.mm_projector') or k.startswith('vision_proj.')]
if len(mm_tensors) == 0:
    if last_checkpoint is not None:
        for k, v in last_checkpoint.items():
            print(k)
    print(f'Found {len(mm_tensors)} tensors to extract out of {len(last_checkpoint)} tensors.')
    print('No tensors found. Is this a LLaVA model?')
    exit()
print(f'Found {len(mm_tensors)} tensors to extract.')
print(f'Found additional {len(first_mm_tensors)} tensors to extract.')
projector = {}
for name in mm_tensors:
    projector[name] = last_checkpoint[name].float()
for name in first_mm_tensors:
    projector[name] = first_checkpoint[name].float()
if len(projector) > 0:
    save_model(projector, f'{args.model}/llava.projector', 'pytorch')
print('Done!')
print(f'Now you can convert {args.model} to a a regular LLaMA GGUF file.')
print(f'Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.')