"""
Hierarchical Diffusion Architecture
Novel cascading diffusion framework for modeling distributions of distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Temporal positional encoding for diffusion timesteps"""
    
    def __init__(self, dim: int, max_seq_len: int = 10000):
        super().__init__()
        self.dim = dim
        
        # Precompute positional encodings
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                            (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timestep indices [batch_size]
        Returns:
            positional encoding [batch_size, dim]
        """
        return self.pe[0, t]


class DiffusionBlock(nn.Module):
    """
    Single-level diffusion block that models p(x_t | x_{t-1})
    Learns to denoise from noise to data distribution
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main denoising network
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy data [batch_size, input_dim]
            t_emb: time embedding [batch_size, time_dim]
        Returns:
            denoised prediction [batch_size, input_dim]
        """
        t_proj = self.time_proj(t_emb)
        x_t_proj = torch.cat([x, t_proj], dim=-1)
        return self.net(x_t_proj)


class HierarchicalDiffusionBlock(nn.Module):
    """
    Hierarchical Diffusion Block: Models p(z_l | z_{l-1})
    
    Core innovation: Cascading diffusion where each level models
    the distribution of the previous level's latent space.
    
    This implements: diffusion(diffusion(x)) -> distribution of distributions
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 time_dim: int,
                 num_levels: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_levels = num_levels
        
        # Positional encoding for timesteps
        self.pos_encoding = PositionalEncoding(time_dim)
        
        # Cascade of diffusion blocks
        # Level 0: models p(x | noise) - base data distribution
        # Level 1: models p(z_1 | noise) - distribution of latent codes
        # Level 2+: models p(z_l | noise) - distribution of distributions
        self.diffusion_levels = nn.ModuleList([
            DiffusionBlock(input_dim, hidden_dim, time_dim)
            for _ in range(num_levels)
        ])
        
        # Latent projection between levels
        self.level_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_levels - 1)
        ])
        
        # Variance schedule (learned or fixed)
        self.register_buffer('betas', self._linear_beta_schedule(1000))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def _linear_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """Linear variance schedule"""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,
                level: int = 0) -> torch.Tensor:
        """
        Forward pass through hierarchical diffusion
        
        Args:
            x: input data [batch_size, input_dim]
            t: timestep [batch_size]
            level: which diffusion level to use (0 to num_levels-1)
        
        Returns:
            denoised output [batch_size, input_dim]
        """
        if level >= self.num_levels:
            raise ValueError(f"Level {level} exceeds num_levels {self.num_levels}")
        
        # Get time embedding
        t_emb = self.pos_encoding(t)
        
        # Apply diffusion at specified level
        output = self.diffusion_levels[level](x, t_emb)
        
        return output
    
    def cascade_forward(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Cascade through all hierarchical levels
        Returns latent representations at each level
        
        Args:
            x: input data [batch_size, input_dim]
            t: timestep [batch_size]
        
        Returns:
            dict with keys 'level_0', 'level_1', ... containing
            latent representations at each hierarchical level
        """
        outputs = {}
        current = x
        
        for level in range(self.num_levels):
            # Denoise at current level
            denoised = self.forward(current, t, level=level)
            outputs[f'level_{level}'] = denoised
            
            # Project to next level's latent space
            if level < self.num_levels - 1:
                current = self.level_projections[level](denoised)
        
        return outputs
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise according to variance schedule
        
        Args:
            x: clean data [batch_size, input_dim]
            t: timestep [batch_size]
        
        Returns:
            (noisy_x, noise) tuple
        """
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1)
        
        noise = torch.randn_like(x)
        noisy_x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        
        return noisy_x, noise


class MultiScaleHierarchicalDiffusion(nn.Module):
    """
    Multi-scale hierarchical diffusion for handling multiple modalities
    Each modality has its own hierarchical diffusion cascade
    """
    
    def __init__(self,
                 modality_dims: Dict[str, int],
                 hidden_dim: int,
                 time_dim: int,
                 num_levels: int = 2):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_levels = num_levels
        
        # Create hierarchical diffusion for each modality
        self.modality_diffusions = nn.ModuleDict({
            modality: HierarchicalDiffusionBlock(
                input_dim=dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                num_levels=num_levels
            )
            for modality, dim in modality_dims.items()
        })
        
        # Cross-modality attention for information flow
        self.cross_modality_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, 
                x_dict: Dict[str, torch.Tensor],
                t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modality hierarchical diffusion
        
        Args:
            x_dict: dict mapping modality names to tensors [batch_size, modality_dim]
            t: timestep [batch_size]
        
        Returns:
            dict mapping modality names to denoised outputs
        """
        outputs = {}
        
        for modality, x in x_dict.items():
            outputs[modality] = self.modality_diffusions[modality].cascade_forward(x, t)
        
        return outputs


if __name__ == "__main__":
    # Test hierarchical diffusion
    batch_size = 4
    input_dim = 32
    hidden_dim = 64
    time_dim = 128
    
    # Create model
    model = HierarchicalDiffusionBlock(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_levels=3
    )
    
    # Test data
    x = torch.randn(batch_size, input_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = model.forward(x, t, level=0)
    print(f"Single level output shape: {output.shape}")
    
    # Cascade forward
    cascade_outputs = model.cascade_forward(x, t)
    print(f"Cascade outputs: {cascade_outputs.keys()}")
    for key, val in cascade_outputs.items():
        print(f"  {key}: {val.shape}")
