"""
Sub-World C: Strategy & Risk Agent Layer
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from hierarchical_diffusion import MultiScaleHierarchicalDiffusion


class SubWorldC_StrategyAgent(nn.Module):
    
    def __init__(self, hidden_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        modality_dims = {
            'action': 12,
            'risk': 6,
            'objective': 4
        }
        
        self.diffusion = MultiScaleHierarchicalDiffusion(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_levels=2
        )
        
        # Project z_A and z_B to feature dimension
        self.z_A_projector = nn.Linear(hidden_dim, sum(modality_dims.values()))
        self.z_B_projector = nn.Linear(hidden_dim, sum(modality_dims.values()))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(sum(modality_dims.values()) * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Risk constraint module
        self.risk_constraint = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Objective alignment module
        self.objective_alignment = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
                z_A: torch.Tensor,
                z_B: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        
        denoised = self.diffusion(x_dict, t)
        
        all_features = torch.cat([
            denoised['action']['level_0'],
            denoised['risk']['level_0'],
            denoised['objective']['level_0']
        ], dim=-1)
        
        z_A_proj = self.z_A_projector(z_A)
        z_B_proj = self.z_B_projector(z_B)
        
        combined_features = torch.cat([all_features, z_A_proj, z_B_proj], dim=-1)
        fused = self.fusion(combined_features)
        
        risk_scale = self.risk_constraint(torch.cat([z_A, z_B], dim=-1))
        objective_embed = self.objective_alignment(torch.cat([z_A, z_B], dim=-1))
        
        latent_state = self.state_encoder(fused)
        
        auxiliary = {
            'risk_scale': risk_scale,
            'objective_embed': objective_embed
        }
        
        return denoised, latent_state, auxiliary
