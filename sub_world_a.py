"""
Sub-World A: Market Microstructure Layer
Models: price paths, volume flows, order imbalance, volatility evolution
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from hierarchical_diffusion import MultiScaleHierarchicalDiffusion


class SubWorldA_MicroStructure(nn.Module):
    """
    Sub-World A: Market Microstructure Layer
    Latent state: z_A = [price_dynamics, volume_flow, order_imbalance, volatility]
    """
    
    def __init__(self, hidden_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        modality_dims = {
            'price': 16,
            'volume': 8,
            'order_imbalance': 4,
            'volatility': 4
        }
        
        self.diffusion = MultiScaleHierarchicalDiffusion(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_levels=2
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(sum(modality_dims.values()), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                x_dict: Dict[str, torch.Tensor],
                t: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        denoised = self.diffusion(x_dict, t)
        
        all_features = torch.cat([
            denoised['price']['level_0'],
            denoised['volume']['level_0'],
            denoised['order_imbalance']['level_0'],
            denoised['volatility']['level_0']
        ], dim=-1)
        
        latent_state = self.state_encoder(all_features)
        return denoised, latent_state
