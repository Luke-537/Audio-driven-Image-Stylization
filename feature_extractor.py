import torch
import torch.nn as nn
from imagebind.models import imagebind_model
import torchvision.transforms as transforms
from imagebind import data as ib_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from PIL import Image
import tqdm
from pathlib import Path
import json
import pickle

import sys
sys.path.insert(1, 'pytorch-AdaIN')
import net


class FeatureExtractor:
    """
    Extracts audio embeddings and style statistics on-the-fly.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        print("Loading models for feature extraction...")
        
        # ImageBind for audio embeddings
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        self.imagebind.to(device)
        
        # VGG for style statistics
        self.vgg_encoder = net.vgg
        self.vgg_encoder.eval()
        self.vgg_encoder.load_state_dict(torch.load(".checkpoints/vgg_normalised.pth"))
        self.vgg_encoder = nn.Sequential(*list(self.vgg_encoder.children())[:31])
        self.vgg_encoder.to(device)
        
        # Image transform for VGG
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("Feature extraction models loaded")
        
    
    def extract_audio_embedding(self, audio_path):
        """
        Extract ImageBind embedding from audio file.
        """
        try:
            inputs = {
                ModalityType.AUDIO: ib_data.load_and_transform_audio_data(
                    [audio_path], self.device
                )
            }
            
            with torch.no_grad():
                embeddings = self.imagebind(inputs)
                audio_embed = embeddings[ModalityType.AUDIO]
            
            return audio_embed.squeeze(0)  # [1024]
            
        except Exception as e:
            print(f"Error extracting audio from {audio_path}: {e}")
            return None
    
    def extract_style_statistics(self, image_path):
        """
        Extract style statistics (mean, std) from style image using VGG.
        """
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.vgg_encoder(image_tensor)  # [1, 512, H, W]
            
            # Compute statistics
            style_mean = features.mean(dim=[2, 3]).squeeze(0)  # [512]
            style_std = features.std(dim=[2, 3]).squeeze(0)    # [512]
            
            return style_mean, style_std
            
        except Exception as e:
            print(f"Error extracting style from {image_path}: {e}")
            return None, None
        


def precompute_all_features(json_paths, output_dir, device='cuda'):
    """
    Precompute embeddings for all datasets.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    print("Loading feature extractor...")
    feature_extractor = FeatureExtractor(device=device)
    
    for split, json_path in json_paths.items():
        print(f"\nProcessing {split} set...")
        
        # Load dataset
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Storage for precomputed features
        precomputed_data = []
        failed_samples = 0
        
        for sample in tqdm.tqdm(data, desc=f"Precomputing {split}"):
            try:
                # Extract audio embedding
                audio_embed = feature_extractor.extract_audio_embedding(
                    sample['audio_path']
                )
                
                # Extract style statistics
                style_mean, style_std = feature_extractor.extract_style_statistics(
                    sample['image_path']
                )
                
                if audio_embed is None or style_mean is None:
                    failed_samples += 1
                    continue
                
                # Convert to CPU tensors for storage
                precomputed_data.append({
                    'audio_embedding': audio_embed.cpu(),
                    'style_mean': style_mean.cpu(),
                    'style_std': style_std.cpu(),
                    'similarity': sample['similarity'],
                    'audio_path': sample['audio_path'],
                    'image_path': sample['image_path']
                })
                
            except Exception as e:
                print(f"Error processing {sample['audio_path']}: {e}")
                failed_samples += 1
        
        # Save precomputed features
        output_path = output_dir / f'{split}_features.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(precomputed_data, f)
        
        print(f"  Saved {len(precomputed_data)} samples to {output_path}")
        print(f"  Failed samples: {failed_samples}")

if __name__ == '__main__':

    precompute_all_features(
        json_paths={
            'train': './datasets/train.json',
            'val': './datasets/val.json'
        },
        output_dir='./datasets',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    