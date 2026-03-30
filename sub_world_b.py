"""
Sub-World B: Macro & Regime Dynamics Layer
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from hierarchical_diffusion import MultiScaleHierarchicalDiffusion


class SubWorldB_MacroRegime(nn.Module):
    
    def __init__(self, hidden_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        modality_dims = {
            'regime': 8,
            'vol_structure': 8,
            'liquidity': 4,
            'risk_appetite': 4
        }
        
        self.diffusion = MultiScaleHierarchicalDiffusion(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_levels=2
        )
        
        self.z_A_projector = nn.Linear(hidden_dim, sum(modality_dims.values()))
        
        self.fusion = nn.Sequential(
            nn.Linear(sum(modality_dims.values()) * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                t: torch.Tensor,
                z_A: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        denoised = self.diffusion(x_dict, t)
        
        all_features = torch.cat([
            denoised['regime']['level_0'],
            denoised['vol_structure']['level_0'],
            denoised['liquidity']['level_0'],
            denoised['risk_appetite']['level_0']
        ], dim=-1)
        
        if z_A is not None:
            z_A_proj = self.z_A_projector(z_A)
            combined_features = torch.cat([all_features, z_A_proj], dim=-1)
            fused = self.fusion(combined_features)
        else:
            fused = all_features
        
        latent_state = self.state_encoder(fused)
        
        return denoised, latent_state
