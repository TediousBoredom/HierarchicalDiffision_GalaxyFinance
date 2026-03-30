"""
Advanced Usage Patterns for Hierarchical Diffusion World Model
"""

import torch
from financial_world_model import HierarchicalDiffusionWorldFinance
from trainer import HierarchicalWorldModelTrainer


def advanced_training_loop():
    """Advanced training with validation and monitoring"""
    print("=" * 60)
    print("PATTERN 1: Advanced Training Loop")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    trainer = HierarchicalWorldModelTrainer(model, device=device)
    
    num_epochs = 3
    batch_size = 4
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': 0.0, 'A': 0.0, 'B': 0.0, 'C': 0.0}
        
        for batch in range(3):
            x_A = {
                'price': torch.randn(batch_size, 16, device=device),
                'volume': torch.randn(batch_size, 8, device=device),
                'order_imbalance': torch.randn(batch_size, 4, device=device),
                'volatility': torch.randn(batch_size, 4, device=device)
            }
            
            x_B = {
                'regime': torch.randn(batch_size, 8, device=device),
                'vol_structure': torch.randn(batch_size, 8, device=device),
                'liquidity': torch.randn(batch_size, 4, device=device),
                'risk_appetite': torch.randn(batch_size, 4, device=device)
            }
            
            x_C = {
                'action': torch.randn(batch_size, 12, device=device),
                'risk': torch.randn(batch_size, 6, device=device),
                'objective': torch.randn(batch_size, 4, device=device)
            }
            
            noise_A = {k: torch.randn_like(v) for k, v in x_A.items()}
            noise_B = {k: torch.randn_like(v) for k, v in x_B.items()}
            noise_C = {k: torch.randn_like(v) for k, v in x_C.items()}
            
            t = torch.randint(0, 1000, (batch_size,), device=device)
            
            losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
            epoch_losses['total'] += losses['total_loss']
            epoch_losses['A'] += losses['loss_A']
            epoch_losses['B'] += losses['loss_B']
            epoch_losses['C'] += losses['loss_C']
        
        for key in epoch_losses:
            epoch_losses[key] /= 3
        
        print(f"Epoch {epoch + 1}: Total={epoch_losses['total']:.4f}")
        trainer.step_scheduler()
    
    print("✓ Advanced training complete\n")


def conditional_generation():
    """Generate trajectories with specific constraints"""
    print("=" * 60)
    print("PATTERN 2: Conditional Generation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    
    print("\nScenario 1: High Volatility Regime")
    x_A = {
        'price': torch.randn(batch_size, 16, device=device) * 2.0,
        'volume': torch.randn(batch_size, 8, device=device) * 1.5,
        'order_imbalance': torch.randn(batch_size, 4, device=device),
        'volatility': torch.ones(batch_size, 4, device=device) * 0.8
    }
    
    x_B = {
        'regime': torch.ones(batch_size, 8, device=device) * 0.5,
        'vol_structure': torch.ones(batch_size, 8, device=device) * 0.7,
        'liquidity': torch.ones(batch_size, 4, device=device) * 0.3,
        'risk_appetite': torch.ones(batch_size, 4, device=device) * 0.2
    }
    
    x_C = {
        'action': torch.randn(batch_size, 12, device=device),
        'risk': torch.ones(batch_size, 6, device=device) * 0.9,
        'objective': torch.ones(batch_size, 4, device=device) * 0.5
    }
    
    t = torch.tensor([500, 500], device=device)
    
    with torch.no_grad():
        outputs = model(x_A, x_B, x_C, t)
    
    print(f"  Risk scale: {outputs['auxiliary_C']['risk_scale'].mean().item():.4f}")
    
    print("\nScenario 2: Low Volatility Regime")
    x_A['volatility'] = torch.ones(batch_size, 4, device=device) * 0.1
    x_B['regime'] = torch.zeros(batch_size, 8, device=device)
    x_B['liquidity'] = torch.ones(batch_size, 4, device=device) * 0.8
    
    with torch.no_grad():
        outputs = model(x_A, x_B, x_C, t)
    
    print(f"  Risk scale: {outputs['auxiliary_C']['risk_scale'].mean().item():.4f}")
    print("✓ Conditional generation complete\n")


def latent_interpolation():
    """Interpolate between two states in latent space"""
    print("=" * 60)
    print("PATTERN 3: Latent Space Interpolation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    model = model.to(device)
    model.eval()
    
    x_A_1 = {
        'price': torch.randn(1, 16, device=device),
        'volume': torch.randn(1, 8, device=device),
        'order_imbalance': torch.randn(1, 4, device=device),
        'volatility': torch.randn(1, 4, device=device)
    }
    
    x_B_1 = {
        'regime': torch.randn(1, 8, device=device),
        'vol_structure': torch.randn(1, 8, device=device),
        'liquidity': torch.randn(1, 4, device=device),
        'risk_appetite': torch.randn(1, 4, device=device)
    }
    
    x_C_1 = {
        'action': torch.randn(1, 12, device=device),
        'risk': torch.randn(1, 6, device=device),
        'objective': torch.randn(1, 4, device=device)
    }
    
    x_A_2 = {k: v + torch.randn_like(v) * 0.5 for k, v in x_A_1.items()}
    x_B_2 = {k: v + torch.randn_like(v) * 0.5 for k, v in x_B_1.items()}
    x_C_2 = {k: v + torch.randn_like(v) * 0.5 for k, v in x_C_1.items()}
    
    t = torch.tensor([500], device=device)
    
    with torch.no_grad():
        outputs_1 = model(x_A_1, x_B_1, x_C_1, t)
        outputs_2 = model(x_A_2, x_B_2, x_C_2, t)
    
    z_A_1, z_B_1, z_C_1 = outputs_1['z_A'], outputs_1['z_B'], outputs_1['z_C']
    z_A_2, z_B_2, z_C_2 = outputs_2['z_A'], outputs_2['z_B'], outputs_2['z_C']
    
    print("\nInterpolating between two market states:")
    for i in range(5):
        alpha = i / 4.0
        z_A_interp = (1 - alpha) * z_A_1 + alpha * z_A_2
        z_B_interp = (1 - alpha) * z_B_1 + alpha * z_B_2
        z_C_interp = (1 - alpha) * z_C_1 + alpha * z_C_2
        
        print(f"  Step {i}: α={alpha:.2f}, ||z_A||={z_A_interp.norm().item():.4f}")
    
    print("✓ Latent interpolation complete\n")


def multi_scale_analysis():
    """Analyze model behavior at different diffusion scales"""
    print("=" * 60)
    print("PATTERN 4: Multi-Scale Analysis")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    
    x_A = {
        'price': torch.randn(batch_size, 16, device=device),
        'volume': torch.randn(batch_size, 8, device=device),
        'order_imbalance': torch.randn(batch_size, 4, device=device),
        'volatility': torch.randn(batch_size, 4, device=device)
    }
    
    x_B = {
        'regime': torch.randn(batch_size, 8, device=device),
        'vol_structure': torch.randn(batch_size, 8, device=device),
        'liquidity': torch.randn(batch_size, 4, device=device),
        'risk_appetite': torch.randn(batch_size, 4, device=device)
    }
    
    x_C = {
        'action': torch.randn(batch_size, 12, device=device),
        'risk': torch.randn(batch_size, 6, device=device),
        'objective': torch.randn(batch_size, 4, device=device)
    }
    
    print("\nAnalyzing latent states at different timesteps:")
    for t_val in [100, 300, 500, 700, 900]:
        t = torch.tensor([t_val, t_val], device=device)
        
        with torch.no_grad():
            outputs = model(x_A, x_B, x_C, t)
        
        z_A_norm = outputs['z_A'].norm(dim=1).mean().item()
        z_B_norm = outputs['z_B'].norm(dim=1).mean().item()
        z_C_norm = outputs['z_C'].norm(dim=1).mean().item()
        
        print(f"  t={t_val:3d}: ||z_A||={z_A_norm:.4f}, ||z_B||={z_B_norm:.4f}, ||z_C||={z_C_norm:.4f}")
    
    print("✓ Multi-scale analysis complete\n")


def causal_analysis():
    """Analyze causal dependencies between sub-worlds"""
    print("=" * 60)
    print("PATTERN 5: Causal Dependency Analysis")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    model = model.to(device)
    model.eval()
    
    batch_size = 4
    
    x_A = {
        'price': torch.randn(batch_size, 16, device=device),
        'volume': torch.randn(batch_size, 8, device=device),
        'order_imbalance': torch.randn(batch_size, 4, device=device),
        'volatility': torch.randn(batch_size, 4, device=device)
    }
    
    x_B = {
        'regime': torch.randn(batch_size, 8, device=device),
        'vol_structure': torch.randn(batch_size, 8, device=device),
        'liquidity': torch.randn(batch_size, 4, device=device),
        'risk_appetite': torch.randn(batch_size, 4, device=device)
    }
    
    x_C = {
        'action': torch.randn(batch_size, 12, device=device),
        'risk': torch.randn(batch_size, 6, device=device),
        'objective': torch.randn(batch_size, 4, device=device)
    }
    
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    with torch.no_grad():
        outputs = model(x_A, x_B, x_C, t)
    
    z_A = outputs['z_A']
    z_B = outputs['z_B']
    z_C = outputs['z_C']
    
    def correlation(x, y):
        x_norm = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        y_norm = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-8)
        return (x_norm * y_norm).mean().item()
    
    corr_A_B = correlation(z_A, z_B)
    corr_A_C = correlation(z_A, z_C)
    corr_B_C = correlation(z_B, z_C)
    
    print("\nCausal Correlations:")
    print(f"  z_A → z_B: {corr_A_B:.4f}")
    print(f"  z_A → z_C: {corr_A_C:.4f}")
    print(f"  z_B → z_C: {corr_B_C:.4f}")
    print("✓ Causal analysis complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Advanced Usage Patterns")
    print("=" * 60 + "\n")
    
    advanced_training_loop()
    conditional_generation()
    latent_interpolation()
    multi_scale_analysis()
    causal_analysis()
    
    print("=" * 60)
    print("All advanced patterns completed!")
    print("=" * 60 + "\n")
