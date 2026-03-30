"""
Hierarchical Diffusion World Model for Financial Markets
Complete three-layer architecture with causal boundaries
"""

import torch
import torch.nn as nn
from typing import Dict
from sub_world_a import SubWorldA_MicroStructure
from sub_world_b import SubWorldB_MacroRegime
from sub_world_c import SubWorldC_StrategyAgent


class HierarchicalDiffusionWorldFinance(nn.Module):
    """
    Complete Hierarchical Diffusion World Model for Financial Markets
    
    Three-layer architecture with causal boundaries:
    - Sub-World A: Market Microstructure (independent)
    - Sub-World B: Macro & Regime (reads A)
    - Sub-World C: Strategy Agent (reads A and B)
    """
    
    def __init__(self, hidden_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        self.sub_world_A = SubWorldA_MicroStructure(hidden_dim, time_dim)
        self.sub_world_B = SubWorldB_MacroRegime(hidden_dim, time_dim)
        self.sub_world_C = SubWorldC_StrategyAgent(hidden_dim, time_dim)
        
        self.global_state_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self,
                x_A: Dict[str, torch.Tensor],
                x_B: Dict[str, torch.Tensor],
                x_C: Dict[str, torch.Tensor],
                t: torch.Tensor) -> Dict:
        """
        Forward pass through hierarchical world model
        
        Args:
            x_A: microstructure modalities
            x_B: macro regime modalities
            x_C: strategy modalities
            t: timestep
        
        Returns:
            dict with all outputs and latent states
        """
        # Sub-World A: independent
        denoised_A, z_A = self.sub_world_A(x_A, t)
        
        # Sub-World B: reads z_A
        denoised_B, z_B = self.sub_world_B(x_B, t, z_A=z_A)
        
        # Sub-World C: reads z_A and z_B
        denoised_C, z_C, aux_C = self.sub_world_C(x_C, t, z_A=z_A, z_B=z_B)
        
        # Global state
        global_state = self.global_state_aggregator(
            torch.cat([z_A, z_B, z_C], dim=-1)
        )
        
        return {
            'denoised_A': denoised_A,
            'denoised_B': denoised_B,
            'denoised_C': denoised_C,
            'z_A': z_A,
            'z_B': z_B,
            'z_C': z_C,
            'global_state': global_state,
            'auxiliary_C': aux_C
        }


if __name__ == "__main__":
    batch_size = 4
    hidden_dim = 64
    time_dim = 128
    
    model = HierarchicalDiffusionWorldFinance(hidden_dim, time_dim)
    
    x_A = {
        'price': torch.randn(batch_size, 16),
        'volume': torch.randn(batch_size, 8),
        'order_imbalance': torch.randn(batch_size, 4),
        'volatility': torch.randn(batch_size, 4)
    }
    
    x_B = {
        'regime': torch.randn(batch_size, 8),
        'vol_structure': torch.randn(batch_size, 8),
        'liquidity': torch.randn(batch_size, 4),
        'risk_appetite': torch.randn(batch_size, 4)
    }
    
    x_C = {
        'action': torch.randn(batch_size, 12),
        'risk': torch.randn(batch_size, 6),
        'objective': torch.randn(batch_size, 4)
    }
    
    t = torch.randint(0, 1000, (batch_size,))
    
    outputs = model(x_A, x_B, x_C, t)
    
    print("Model outputs:")
    for key in outputs:
        if isinstance(outputs[key], torch.Tensor):
            print(f"  {key}: {outputs[key].shape}")
        elif isinstance(outputs[key], dict):
            print(f"  {key}: dict")
