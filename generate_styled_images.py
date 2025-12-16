import json
import random
import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torchvision.utils import save_image
import sys
sys.path.insert(1, 'pytorch-AdaIN')
import net
from function import adaptive_instance_normalization


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def process_json_with_stylization(input_json_path, output_json_path, output_dir, device):
    """
    Load JSON, stylize content images for each entry, save paths back to new JSON.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        output_dir: Directory to save stylized images
        device: torch device (cuda/cpu)
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load models
    print("Loading AdaIN models...")
    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load('.checkpoints/decoder.pth'))
    vgg.load_state_dict(torch.load('.checkpoints/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    decoder.to(device)

    print(f"VGG is on: {next(vgg.parameters()).device}")
    print(f"Decoder is on: {next(decoder.parameters()).device}")
    
    # Setup transforms
    content_tf = test_transform(512, crop=False)
    style_tf = test_transform(512, crop=False)
    
    # Load JSON data
    print(f"Loading JSON from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    # Process each entry
    for idx, entry in enumerate(data):
        entry['stylized_images'] = []
        
        # Get the style image path from the entry
        style_path = Path(entry['image_path'])
        
        if not style_path.exists():
            print(f"Warning: Style image not found: {style_path}")
            continue
        
        # Load style image once
        try:
            style_img = Image.open(str(style_path)).convert('RGB')
            style = style_tf(style_img)
            style = style.to(device).unsqueeze(0)
        except Exception as e:
            print(f"Error loading style image {style_path}: {e}")
            continue
        
        # Stylize each content image with this style
        for content_file in entry['random_images']:
            content_path = Path(content_file)
            
            try:
                # Load content image (convert to RGB to handle RGBA)
                content_img = Image.open(str(content_path)).convert('RGB')
                content = content_tf(content_img)
                content = content.to(device).unsqueeze(0)
                
                # Perform style transfer
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style, alpha=1.0)
                    output = output.cpu()
                
                # Save stylized image
                output_name = output_dir / f'{content_path.stem}_stylized_{style_path.stem}.jpg'
                save_image(output, str(output_name))
                
                # Store path in entry
                entry['stylized_images'].append(str(output_name))
                
            except Exception as e:
                print(f"Error processing {content_file} with {style_path.name}: {e}")
                continue
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(data)} entries")
            #break
    
    # Save updated JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Complete! Saved to {output_json_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_json = "audio_style_content_pairs.json"
    output_json = "stylized.json"
    output_directory = "/graphics/scratch2/students/reutemann/stylized_images"
    
    process_json_with_stylization(
        input_json, 
        output_json, 
        output_directory, 
        device
    )