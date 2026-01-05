import torch
from torch.utils.data import Dataset
import pickle


class AudioStyleDataset(Dataset):
    """
    Dataset that loads pre-computed embeddings.
    """
    def __init__(self, pickle_path, normalize=False):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} precomputed samples from {pickle_path}")
        
        self.normalize = normalize
        
        if normalize:
            # Compute normalization statistics separately for means and stds
            all_means = torch.stack([s['style_mean'] for s in self.data])
            all_stds = torch.stack([s['style_std'] for s in self.data])
            
            # For means: use mean and std
            self.mean_mean = all_means.mean(dim=0)  # [512]
            self.mean_std = all_means.std(dim=0)    # [512]
            
            # For stds: use mean and std (stds are always positive)
            self.std_mean = all_stds.mean(dim=0)    # [512]
            self.std_std = all_stds.std(dim=0)      # [512]
            
            # Normalize each sample
            for i in range(len(self.data)):
                self.data[i]['style_mean'] = (
                    (self.data[i]['style_mean'] - self.mean_mean) / (self.mean_std + 1e-8)
                )
                self.data[i]['style_std'] = (
                    (self.data[i]['style_std'] - self.std_mean) / (self.std_std + 1e-8)
                )
            
            print(f"Normalization stats computed:")
            print(f"  Mean - mean: {self.mean_mean.mean().item():.4f}, std: {self.mean_std.mean().item():.4f}")
            print(f"  Std  - mean: {self.std_mean.mean().item():.4f}, std: {self.std_std.mean().item():.4f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return {
            'audio_embedding': sample['audio_embedding'],
            'style_mean': sample['style_mean'],
            'style_std': sample['style_std'],
        }
    