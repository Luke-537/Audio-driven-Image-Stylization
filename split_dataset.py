import json
import numpy as np
from pathlib import Path


def split_dataset(input_json, output_dir, train_ratio=0.9,seed=42):
    """
    Split the matches JSON into train and validation sets.
    
    Args:
        input_json: Path to your matches.json file
        output_dir: Where to save train.json and val.json
        train_ratio: Proportion for training (0.9 = 90% train, 10% val)
        seed: Random seed for reproducibility
    """
    
    # Load matches
    print(f"\nLoading matches from {input_json}")
    with open(input_json, 'r') as f:
        matches = json.load(f)
    
    print(f"Total matches: {len(matches)}")
    
    # Shuffle
    np.random.seed(seed)
    indices = np.random.permutation(len(matches))
    matches = [matches[i] for i in indices]
    
    # Split
    split_idx = int(len(matches) * train_ratio)
    train_data = matches[:split_idx]
    val_data = matches[split_idx:]
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_path / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Save metadata
    metadata = {
        'total_samples': len(matches),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'train_ratio': train_ratio,
        'seed': seed
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Train: {len(train_data)} samples ({train_ratio*100:.1f}%)")
    print(f"   Val: {len(val_data)} samples ({(1-train_ratio)*100:.1f}%)")
    print(f"   Saved to: {output_path}")
    
if __name__ == "__main__":
    split_dataset(
        input_json='data/audio_image_pairs.json',
        output_dir='data',
        train_ratio=0.9,
        seed=3
    )
