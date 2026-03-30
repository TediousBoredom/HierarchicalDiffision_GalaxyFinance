"""
Example usage and demonstration of Hierarchical Diffusion World Model
"""

import torch
from financial_world_model import HierarchicalDiffusionWorldFinance
from trainer import HierarchicalWorldModelTrainer, DiffusionSampler


def demo_forward_pass():
    """Demonstrate forward pass through the model"""
    print("=" * 60)
    print("DEMO 1: Forward Pass Through Hierarchical World Model")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    
    # Initialize model
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    model = model.to(device)
    
    # Create dummy inputs
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
    
    # Forward pass
    outputs = model(x_A, x_B, x_C, t)
    
    print("\nInput shapes:")
    print(f"  Sub-World A (Microstructure):")
    for k, v in x_A.items():
        print(f"    {k}: {v.shape}")
    print(f"  Sub-World B (Macro & Regime):")
    for k, v in x_B.items():
        print(f"    {k}: {v.shape}")
    print(f"  Sub-World C (Strategy Agent):")
    for k, v in x_C.items():
        print(f"    {k}: {v.shape}")
    
    print("\nOutput latent states:")
    print(f"  z_A (Microstructure latent): {outputs['z_A'].shape}")
    print(f"  z_B (Macro latent): {outputs['z_B'].shape}")
    print(f"  z_C (Strategy latent): {outputs['z_C'].shape}")
    print(f"  Global state: {outputs['global_state'].shape}")
    
    print("\nCausal information flow:")
    print("  Sub-World A → (independent)")
    print("  Sub-World B → (reads z_A)")
    print("  Sub-World C → (reads z_A, z_B)")
    
    print("\n✓ Forward pass successful!")


def demo_training():
    """Demonstrate training loop"""
    print("\n" + "=" * 60)
    print("DEMO 2: Training Loop")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    num_steps = 5
    
    # Initialize model and trainer
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    trainer = HierarchicalWorldModelTrainer(model, learning_rate=1e-4, device=device)
    
    print(f"\nTraining on {device}")
    print(f"Batch size: {batch_size}, Steps: {num_steps}\n")
    
    for step in range(num_steps):
        # Create random batch
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
        
        noise_A = {k: torch.randn_like(v) for k, v in x_A.items()}
        noise_B = {k: torch.randn_like(v) for k, v in x_B.items()}
        noise_C = {k: torch.randn_like(v) for k, v in x_C.items()}
        
        t = torch.randint(0, 1000, (batch_size,))
        
        # Training step
        losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
        
        print(f"Step {step + 1}/{num_steps}")
        print(f"  Total Loss: {losses['total_loss']:.6f}")
        print(f"  Loss A: {losses['loss_A']:.6f}, Loss B: {losses['loss_B']:.6f}, Loss C: {losses['loss_C']:.6f}")
        print(f"  Causal Loss B: {losses['loss_causal_B']:.6f}, Causal Loss C: {losses['loss_causal_C']:.6f}")
    
    print("\n✓ Training loop successful!")


def demo_sampling():
    """Demonstrate sampling from the model"""
    print("\n" + "=" * 60)
    print("DEMO 3: Sampling Trajectories")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model and sampler
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    sampler = DiffusionSampler(model, num_steps=100, device=device)
    
    print(f"\nSampling on {device}")
    print("Generating 2 trajectories of length 5...\n")
    
    trajectories = sampler.sample(batch_size=2, seq_length=5)
    
    print("Sampled trajectory shapes:")
    for key, val in trajectories.items():
        print(f"  {key}: {val.shape}")
        print(f"    (batch_size=2, seq_length=5, latent_dim={val.shape[-1]})")
    
    print("\n✓ Sampling successful!")


def demo_architecture():
    """Print architecture information"""
    print("\n" + "=" * 60)
    print("DEMO 4: Architecture Overview")
    print("=" * 60)
    
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    
    print("\nHierarchical Diffusion World Model Architecture:")
    print("\n1. Sub-World A: Market Microstructure (Independent)")
    print("   Modalities:")
    print("     - Price paths (16 dims)")
    print("     - Volume flows (8 dims)")
    print("     - Order imbalance (4 dims)")
    print("     - Volatility evolution (4 dims)")
    print("   Output: z_A (64 dims latent state)")
    
    print("\n2. Sub-World B: Macro & Regime Dynamics (Reads z_A)")
    print("   Modalities:")
    print("     - Trend/oscillation state (8 dims)")
    print("     - Volatility structure (8 dims)")
    print("     - Liquidity environment (4 dims)")
    print("     - Risk appetite (4 dims)")
    print("   Output: z_B (64 dims latent state)")
    print("   Causal dependency: z_A → z_B")
    
    print("\n3. Sub-World C: Strategy & Risk Agent (Reads z_A, z_B)")
    print("   Modalities:")
    print("     - Trading action distribution (12 dims)")
    print("     - Risk embedding (6 dims)")
    print("     - Objective alignment (4 dims)")
    print("   Output: z_C (64 dims latent state)")
    print("   Causal dependencies: z_A, z_B → z_C")
    
    print("\n4. Global State Aggregator")
    print("   Combines: z_A + z_B + z_C → global_state (64 dims)")
    
    print("\nKey Features:")
    print("  ✓ Hierarchical cascading diffusion blocks")
    print("  ✓ Multi-modal modeling for each sub-world")
    print("  ✓ Causal boundaries between layers")
    print("  ✓ Cross-world attention mechanisms")
    print("  ✓ Risk constraints and objective alignment")
    
    print("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params:,}")
    
    print("\n✓ Architecture overview complete!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Hierarchical Diffusion World Model for Financial Markets")
    print("=" * 60)
    
    demo_architecture()
    demo_forward_pass()
    demo_training()
    demo_sampling()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60 + "\n")
