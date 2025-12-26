import torch
from torch.utils.data import Dataset
import pickle


class AudioStyleDataset(Dataset):
    """
    Dataset that loads pre-computed embeddings.
    """
    def __init__(self, pickle_path, normalize=True):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} precomputed samples from {pickle_path}")
        
        # Convert back to tensors if stored as numpy arrays
        if normalize:
            # Calculate mean and std across dataset
            self.style_mean_all = torch.stack([s['style_mean'] for s in self.data]).mean(dim=0)
            self.style_std_all = torch.stack([s['style_std'] for s in self.data]).std(dim=0)
            
            # Normalize each sample
            for i, sample in enumerate(self.data):
                self.data[i]['style_mean'] = (sample['style_mean'] - self.style_mean_all) / self.style_std_all
                self.data[i]['style_std'] = (sample['style_std'] - self.style_mean_all) / self.style_std_all
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return {
            'audio_embedding': sample['audio_embedding'],
            'style_mean': sample['style_mean'],
            'style_std': sample['style_std'],
        }
    