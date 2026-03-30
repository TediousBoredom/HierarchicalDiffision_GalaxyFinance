"""
Training framework for Hierarchical Diffusion World Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from financial_world_model import HierarchicalDiffusionWorldFinance


class DiffusionLoss(nn.Module):
    """Diffusion loss: MSE between predicted noise and actual noise"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(predicted, target)


class HierarchicalWorldModelTrainer:
    """Trainer for Hierarchical Diffusion World Model"""
    
    def __init__(self, 
                 model: HierarchicalDiffusionWorldFinance,
                 learning_rate: float = 1e-4,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = DiffusionLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
    
    def train_step(self,
                   x_A: Dict[str, torch.Tensor],
                   x_B: Dict[str, torch.Tensor],
                   x_C: Dict[str, torch.Tensor],
                   t: torch.Tensor,
                   noise_A: Dict[str, torch.Tensor],
                   noise_B: Dict[str, torch.Tensor],
                   noise_C: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        outputs = self.model(x_A, x_B, x_C, t)
        
        loss_A = self._compute_subworld_loss(outputs['denoised_A'], noise_A)
        loss_B = self._compute_subworld_loss(outputs['denoised_B'], noise_B)
        loss_C = self._compute_subworld_loss(outputs['denoised_C'], noise_C)
        
        loss_causal_B = self._causal_consistency_loss(
            outputs['z_A'], outputs['z_B']
        )
        
        # For z_C, use average of z_A and z_B as parent
        z_parent_C = (outputs['z_A'] + outputs['z_B']) / 2.0
        loss_causal_C = self._causal_consistency_loss(
            z_parent_C,
            outputs['z_C']
        )
        
        total_loss = loss_A + loss_B + loss_C + 0.1 * (loss_causal_B + loss_causal_C)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss_A': loss_A.item(),
            'loss_B': loss_B.item(),
            'loss_C': loss_C.item(),
            'loss_causal_B': loss_causal_B.item(),
            'loss_causal_C': loss_causal_C.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_subworld_loss(self, 
                               denoised: Dict[str, Dict],
                               target_noise: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a sub-world"""
        loss = 0.0
        for modality in target_noise:
            pred = denoised[modality]['level_0']
            target = target_noise[modality]
            loss += self.loss_fn(pred, target)
        return loss
    
    def _causal_consistency_loss(self, 
                                 parent_state: torch.Tensor,
                                 child_state: torch.Tensor) -> torch.Tensor:
        """Causal consistency: child state should be predictable from parent"""
        parent_norm = torch.nn.functional.normalize(parent_state, dim=-1)
        child_norm = torch.nn.functional.normalize(child_state, dim=-1)
        correlation = torch.mean(parent_norm * child_norm)
        return -correlation
    
    def step_scheduler(self):
        """Update learning rate scheduler"""
        self.scheduler.step()


class DiffusionSampler:
    """Sampler for generating trajectories from the model"""
    
    def __init__(self, 
                 model: HierarchicalDiffusionWorldFinance,
                 num_steps: int = 1000,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.num_steps = num_steps
        self.model.eval()
    
    @torch.no_grad()
    def sample(self, batch_size: int, seq_length: int = 10) -> Dict[str, torch.Tensor]:
        """Sample trajectories from the model"""
        trajectories = {
            'z_A': [],
            'z_B': [],
            'z_C': [],
            'global_state': []
        }
        
        x_A = {
            'price': torch.randn(batch_size, 16, device=self.device),
            'volume': torch.randn(batch_size, 8, device=self.device),
            'order_imbalance': torch.randn(batch_size, 4, device=self.device),
            'volatility': torch.randn(batch_size, 4, device=self.device)
        }
        
        x_B = {
            'regime': torch.randn(batch_size, 8, device=self.device),
            'vol_structure': torch.randn(batch_size, 8, device=self.device),
            'liquidity': torch.randn(batch_size, 4, device=self.device),
            'risk_appetite': torch.randn(batch_size, 4, device=self.device)
        }
        
        x_C = {
            'action': torch.randn(batch_size, 12, device=self.device),
            'risk': torch.randn(batch_size, 6, device=self.device),
            'objective': torch.randn(batch_size, 4, device=self.device)
        }
        
        for step in range(self.num_steps - 1, -1, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            
            outputs = self.model(x_A, x_B, x_C, t)
            
            if step % (self.num_steps // seq_length) == 0:
                trajectories['z_A'].append(outputs['z_A'].cpu())
                trajectories['z_B'].append(outputs['z_B'].cpu())
                trajectories['z_C'].append(outputs['z_C'].cpu())
                trajectories['global_state'].append(outputs['global_state'].cpu())
            
            if step > 0:
                noise_scale = 0.1
                x_A = {k: v + noise_scale * torch.randn_like(v) for k, v in x_A.items()}
                x_B = {k: v + noise_scale * torch.randn_like(v) for k, v in x_B.items()}
                x_C = {k: v + noise_scale * torch.randn_like(v) for k, v in x_C.items()}
        
        for key in trajectories:
            trajectories[key] = torch.stack(trajectories[key], dim=1)
        
        return trajectories
