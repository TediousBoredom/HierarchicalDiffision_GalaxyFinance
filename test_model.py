"""
Quick test of the Hierarchical Diffusion World Model
"""

import torch
from financial_world_model import HierarchicalDiffusionWorldFinance
from trainer import HierarchicalWorldModelTrainer


def test_model():
    print("Testing Hierarchical Diffusion World Model\n")
    
    device = 'cpu'
    batch_size = 2
    
    # Create model
    model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)
    model = model.to(device)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Test forward pass
    print("1. Testing forward pass...")
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
    
    outputs = model(x_A, x_B, x_C, t)
    print(f"   z_A shape: {outputs['z_A'].shape}")
    print(f"   z_B shape: {outputs['z_B'].shape}")
    print(f"   z_C shape: {outputs['z_C'].shape}")
    print(f"   global_state shape: {outputs['global_state'].shape}")
    print("   ✓ Forward pass successful\n")
    
    # Test training step
    print("2. Testing training step...")
    trainer = HierarchicalWorldModelTrainer(model, learning_rate=1e-4, device=device)
    
    noise_A = {k: torch.randn_like(v) for k, v in x_A.items()}
    noise_B = {k: torch.randn_like(v) for k, v in x_B.items()}
    noise_C = {k: torch.randn_like(v) for k, v in x_C.items()}
    
    losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
    print(f"   Total loss: {losses['total_loss']:.6f}")
    print(f"   Loss A: {losses['loss_A']:.6f}")
    print(f"   Loss B: {losses['loss_B']:.6f}")
    print(f"   Loss C: {losses['loss_C']:.6f}")
    print(f"   Causal loss B: {losses['loss_causal_B']:.6f}")
    print(f"   Causal loss C: {losses['loss_causal_C']:.6f}")
    print("   ✓ Training step successful\n")
    
    # Test multiple training steps
    print("3. Testing multiple training steps...")
    for step in range(3):
        x_A = {k: torch.randn_like(v) for k, v in x_A.items()}
        x_B = {k: torch.randn_like(v) for k, v in x_B.items()}
        x_C = {k: torch.randn_like(v) for k, v in x_C.items()}
        noise_A = {k: torch.randn_like(v) for k, v in x_A.items()}
        noise_B = {k: torch.randn_like(v) for k, v in x_B.items()}
        noise_C = {k: torch.randn_like(v) for k, v in x_C.items()}
        
        losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
        print(f"   Step {step+1}: loss={losses['total_loss']:.6f}")
    
    print("   ✓ Multiple training steps successful\n")
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    print("\nArchitecture Summary:")
    print("  Sub-World A: Market Microstructure (independent)")
    print("  Sub-World B: Macro & Regime (reads z_A)")
    print("  Sub-World C: Strategy Agent (reads z_A, z_B)")
    print("\nKey Features:")
    print("  ✓ Hierarchical cascading diffusion blocks")
    print("  ✓ Multi-modal modeling")
    print("  ✓ Causal boundaries between layers")
    print("  ✓ Cross-world information flow")
    print("  ✓ Risk constraints and objective alignment")


if __name__ == "__main__":
    test_model()
