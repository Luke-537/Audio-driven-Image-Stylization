import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

import sys
sys.path.insert(1, 'pytorch-AdaIN')
import net  

from mapping_network import AudioToStyleMapper
from feature_extractor import FeatureExtractor

def load_models(mapper_checkpoint, adaIN_decoder_path, vgg_path, device='cuda'):
    """
    Load all necessary models.
    """

    # Loading trained audio-to-style mapper
    mapper = AudioToStyleMapper().to(device)
    checkpoint = torch.load(mapper_checkpoint, map_location=device)
    mapper.load_state_dict(checkpoint['model_state_dict'])
    mapper.eval()
    
    # Loading AdaIN decoder
    decoder = net.decoder
    decoder.load_state_dict(torch.load(adaIN_decoder_path))
    decoder = decoder.to(device)
    decoder.eval()
    
    # Loading pre-trained VGG 
    vgg = net.vgg
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg = vgg.to(device)
    vgg.eval()
    
    # Loading feature extractor for audio
    feature_extractor = FeatureExtractor(device=device)
    
    return mapper, decoder, vgg, feature_extractor


def extract_style_from_audio(audio_path, mapper, feature_extractor, device='cuda'):
    """
    Extract style statistics from audio using your trained mapper.
    """

    # Extract audio embedding
    with torch.no_grad():
        audio_embed = feature_extractor.extract_audio_embedding(audio_path)
        
        if audio_embed is None:
            raise ValueError(f"Failed to extract audio embedding from {audio_path}")
        
        # Move to device and add batch dimension
        audio_embed = audio_embed.unsqueeze(0).to(device)
        
        # Predict style statistics
        pred_mean, pred_std = mapper(audio_embed)
        
    return pred_mean, pred_std


def apply_style_to_content(content_image, style_mean, style_std, vgg, decoder, alpha=1.0):
    """
    Apply predicted style statistics to content image.
    
    Args:
        content_image: Tensor [1, 3, H, W]
        style_mean: Tensor [1, 512] - predicted mean
        style_std: Tensor [1, 512] - predicted std
        alpha: Style strength (0.0 to 1.0)
    """

    with torch.no_grad():
        # Extract content features from VGG
        content_features = vgg(content_image)  # [1, 512, H, W]
        
        # Get current content statistics
        content_mean = content_features.mean(dim=[2, 3], keepdim=True)  # [1, 512, 1, 1]
        content_std = content_features.std(dim=[2, 3], keepdim=True)    # [1, 512, 1, 1]
        
        # Reshape predicted stats to match feature dimensions
        style_mean = style_mean.unsqueeze(-1).unsqueeze(-1)  # [1, 512, 1, 1]
        style_std = style_std.unsqueeze(-1).unsqueeze(-1)    # [1, 512, 1, 1]
        
        # Apply adaptive instance normalization (ADaIN)
        # Normalize content features
        normalized_content = (content_features - content_mean) / (content_std + 1e-8)
        
        # Apply style statistics
        stylized_features = normalized_content * style_std + style_mean
        
        # Blend with original content based on alpha
        blended_features = stylized_features * alpha + content_features * (1 - alpha)
        
        # Decode to image
        output_image = decoder(blended_features)
    
    return output_image


def test_transform(size, crop):
    """Image transformation for inference."""

    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform



if __name__ == '__main__':

    output_dir = "outputs/images/"
    content_image_path = "outputs/images/woman.png"
    content_audio_path= "outputs/images/vacuum.wav"
    mapper_checkpooint = "/graphics/scratch2/students/reutemann/checkpoints/best_model.pt"
    alpha = 0.8  # Style strength

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    mapper, decoder, vgg, feature_extractor = load_models(
        mapper_checkpooint,
        "./.checkpoints/decoder.pth",
        "./.checkpoints/vgg_normalised.pth",
        device
    )
    
    # Load and transform content image
    print(f"Loading content image: {content_image_path}")
    content_tf = test_transform(512, crop=False)
    content_image = content_tf(Image.open(content_image_path).convert('RGB'))
    content_image = content_image.unsqueeze(0).to(device)
    
    # Extract style from audio
    print(f"Processing audio: {content_audio_path}")
    style_mean, style_std = extract_style_from_audio(
        content_audio_path,
        mapper,
        feature_extractor,
        device
    )
    
    print(f"Predicted style - Mean: {style_mean.mean().item():.3f} ± {style_mean.std().item():.3f}")
    print(f"Predicted style - Std:  {style_std.mean().item():.3f} ± {style_std.std().item():.3f}")
    
    # Apply style transfer
    print(f"Applying style transfer (alpha={alpha})...")
    output = apply_style_to_content(
        content_image,
        style_mean,
        style_std,
        vgg,
        decoder,
        alpha
   )
    
    # Save result
    output_path = output_dir + f"{Path(content_image_path).stem}_styled_by_{Path(content_audio_path).stem}.jpg"
    
    # Convert to PIL and save
    output_cpu = output.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_cpu)
    output_image.save(output_path)
    
    print(f"Saved stylized image to: {output_path}")