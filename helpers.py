import os
import re
import torch

def load_latest_checkpoint(checkpoint_dir, model, optimizer, device='cpu'):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoint found. Starting fresh.")
        return None
    
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_ckpt = checkpoints[-1]
    path = os.path.join(checkpoint_dir, latest_ckpt)
    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint, model, optimizer