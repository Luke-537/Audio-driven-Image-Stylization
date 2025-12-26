import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from pathlib import Path

from audio_style_dataset import AudioStyleDataset
from mapping_network import AudioToStyleMapper


def train(train_features, val_features, checkpoint_dir, num_epochs, batch_size, learning_rate, device):
    """
    Training script using pre-computed embeddings.
    """
    
    print("="*70)
    print("TRAINING AUDIO TO STYLE MAPPING NETWORK")
    print("="*70)
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\n1. Loading precomputed datasets...")
    train_dataset = AudioStyleDataset(train_features)
    val_dataset = AudioStyleDataset(val_features)
    
    # Create dataloaders (no custom collate needed!)
    print("\n2. Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\n3. Creating mapping network...")
    model = AudioToStyleMapper().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training loop
    print("\n4. Starting training...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]')
        for batch in pbar:
            audio_embed = batch['audio_embedding'].to(device)
            target_mean = batch['style_mean'].to(device)
            target_std = batch['style_std'].to(device)
            
            optimizer.zero_grad()
            
            pred_mean, pred_std = model(audio_embed)
            
            loss = F.mse_loss(pred_mean, target_mean) + \
                   F.mse_loss(pred_std, target_std)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss = train_loss / max(train_batches, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]'):
                audio_embed = batch['audio_embedding'].to(device)
                target_mean = batch['style_mean'].to(device)
                target_std = batch['style_std'].to(device)
                
                pred_mean, pred_std = model(audio_embed)
                
                loss = F.mse_loss(pred_mean, target_mean) + \
                       F.mse_loss(pred_std, target_std)
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss = val_loss / max(val_batches, 1)
        
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*70)

if __name__ == '__main__':

    train(
        train_features='./datasets/train_features.pkl',
        val_features='./datasets/val_features.pkl',
        checkpoint_dir='/graphics/scratch2/students/reutemann/checkpoints',
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    