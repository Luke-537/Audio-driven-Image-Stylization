import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioToStyleMapper(nn.Module):
    def __init__(self, audio_dim=1024, style_dim=512, hidden_dims=[4096, 2048, 1024]): # [1024, 1024, 512]
        super().__init__()
        
        layers = []
        in_dim = audio_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Try BatchNorm instead of LayerNorm
                nn.ReLU(inplace=True),
                nn.Dropout(0.2) # 0.3 0.4 0.5
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, style_dim * 2))
        
        self.network = nn.Sequential(*layers)
        self.style_dim = style_dim
    
    def forward(self, audio_embedding):
        params = self.network(audio_embedding)
        mean = params[:, :self.style_dim]
        # Use Softplus for better numerical stability than Exp
        std = F.softplus(params[:, self.style_dim:]) + 1e-5
        #std = torch.exp(params[:, self.style_dim:])
        return mean, std
    