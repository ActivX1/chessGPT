import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import gc
from chess_gpt import ChessGPT, ChessGPTConfig
from chess_tokenizer import load_tokenizer, tokenize_game
from helpers import load_latest_checkpoint

class ChessDataset(Dataset):
    def __init__(self, csv_file, token_to_idx, block_size):
        self.data = pd.read_csv(csv_file)
        self.token_to_idx = token_to_idx
        self.block_size = block_size
# Use 0 as padding index since it's a common practice
        self.pad_idx = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game = self.data.iloc[idx]
        tokens = tokenize_game(game['moves'], game['winner'], self.token_to_idx)
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Create input and target sequences
        x = tokens[:-1]
        y = tokens[1:]
        
        # Handle padding
        if len(x) < self.block_size:
            # Pad sequences to block_size
            padding_length = self.block_size - len(x)
            x = torch.cat([x, torch.full((padding_length,), self.pad_idx, dtype=torch.long)])
            y = torch.cat([y, torch.full((padding_length,), self.pad_idx, dtype=torch.long)])
        else:
            # Truncate if longer than block_size
            x = x[:self.block_size]
            y = y[:self.block_size]

        return x, y

def train():
    # Device configuration
    
    if torch.cuda.is_available():
        device_type = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device_type = "cpu"
        print("Using CPU")
    
    print(f"Using device: {device_type}")

    device = torch.device(device_type)
    
    # Only enable CuDNN benchmark if using GPU
    if device_type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("CUDA optimizations enabled")
    
    # Load tokenizer
    token_to_idx, idx_to_token, _ = load_tokenizer()
    
    # Model configuration
    config = ChessGPTConfig(
        vocab_size=len(token_to_idx),
        block_size=128,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.1
    )
    

    # Training parameters
    num_epochs = 50
    batch_size = 128 if device_type in ['cuda'] else 32
    learning_rate = 3e-4
    accum_iter = 4 if device_type in ['cuda', 'mps'] else 1 # Gradient accumulation steps
    
    # Initialize dataset and dataloader
    dataset = ChessDataset('games.csv', token_to_idx, config.block_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8 if device_type == 'cuda' else 8,  # No workers on CPU
        pin_memory=True if device_type == 'cuda' else False,
        prefetch_factor=2 if device_type == 'cuda' else 2
    )
    
    # Initialize model
    model = ChessGPT(config).to(device)
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Load checkpoint if available
    checkpoint, model, optimizer = load_latest_checkpoint('.', model, optimizer, device=device)
    model = model.to(device)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No .pt checkpoints found in project root. Starting from scratch.")
        start_epoch = 0
   
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1
    )
    
    # Initialize mixed precision only for CUDA
    if device_type == 'cuda':
        scaler = torch.amp.GradScaler()
        print("Mixed precision training enabled")
    
    if device_type in ['cuda']:
        print(f"Compiling model for {device_type.upper()} acceleration...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Model compilation failed: {e}. Continuing with uncompiled model.")
    else:
        print(f"Skipping model compilation on {device_type.upper()} for better performance")

    # Training loop
    step_start = time.time()
    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision on CUDA
            if device_type == 'cuda':
                with torch.amp.autocast():
                    logits, loss = model(x, y)
                    loss = loss / accum_iter
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                # Standard forward and backward pass on CPU
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * (accum_iter if device_type == 'cuda' else 1)
            
            if batch_idx % 100 == 0:
                step_time = time.time() - step_start
                examples_per_sec = batch_size * 100 / step_time if batch_idx > 0 else 0
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.6f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}, '
                      f'Time: {step_time:.2f}s, '
                      f'Examples/sec: {examples_per_sec:.1f}')
                step_start = time.time()  # Reset timer for next 100 steps
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint with device-agnostic loading
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'device': device.type
        }
        if device_type == 'cuda':
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
        del x, y, logits, loss
        gc.collect()
        torch.mps.empty_cache()

if __name__ == '__main__':
    train()