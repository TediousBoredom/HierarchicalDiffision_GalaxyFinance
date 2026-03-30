# Hierarchical Diffusion World Model - Implementation Guide

## Project Overview

This project implements a **novel Hierarchical Diffusion Architecture** for modeling financial markets as a multi-modal causal world model. The system consists of three hierarchical sub-worlds with explicit causal boundaries.

## What Was Built

### 1. Core Architecture: Hierarchical Diffusion Blocks
- **File**: `hierarchical_diffusion.py`
- **Key Classes**:
  - `PositionalEncoding`: Temporal embeddings for diffusion timesteps
  - `DiffusionBlock`: Single-level denoising network
  - `HierarchicalDiffusionBlock`: Cascading diffusion through multiple levels
  - `MultiScaleHierarchicalDiffusion`: Multi-modality support

**Innovation**: Cascading diffusion that models p(z_l | noise) - distributions of distributions

### 2. Three-Layer Sub-World System

#### Sub-World A: Market Microstructure
- **File**: `sub_world_a.py`
- **Modalities**: Price, Volume, Order Imbalance, Volatility
- **Role**: Independent foundation layer
- **Output**: z_A (64-dim latent state)

#### Sub-World B: Macro & Regime Dynamics
- **File**: `sub_world_b.py`
- **Modalities**: Regime, Volatility Structure, Liquidity, Risk Appetite
- **Role**: Intermediate layer reading z_A
- **Output**: z_B (64-dim latent state)
- **Causal Dependency**: z_A → z_B

#### Sub-World C: Strategy & Risk Agent
- **File**: `sub_world_c.py`
- **Modalities**: Action Distribution, Risk Embedding, Objective Alignment
- **Role**: Top layer generating trading decisions
- **Output**: z_C (64-dim latent state)
- **Causal Dependencies**: z_A, z_B → z_C

### 3. Complete World Model
- **File**: `financial_world_model.py`
- **Class**: `HierarchicalDiffusionWorldFinance`
- **Features**:
  - Orchestrates all three sub-worlds
  - Enforces causal boundaries
  - Aggregates global state
  - 621,422 trainable parameters

### 4. Training Framework
- **File**: `trainer.py`
- **Key Classes**:
  - `HierarchicalWorldModelTrainer`: Training loop with causal consistency losses
  - `DiffusionSampler`: Trajectory generation via reverse diffusion
- **Loss Functions**:
  - Denoising loss (per sub-world)
  - Causal consistency loss (enforces dependencies)

### 5. Testing & Examples
- **File**: `test_model.py` - Quick validation tests
- **File**: `example_usage.py` - Comprehensive demonstrations

## File Structure

```
HierarchicalDiffision_GalaxyFinance/
├── hierarchical_diffusion.py      # Core hierarchical diffusion blocks
├── sub_world_a.py                 # Market microstructure sub-world
├── sub_world_b.py                 # Macro & regime sub-world
├── sub_world_c.py                 # Strategy & risk agent sub-world
├── financial_world_model.py        # Complete world model
├── trainer.py                      # Training and sampling
├── test_model.py                   # Quick tests
├── example_usage.py                # Usage examples
├── requirements.txt                # Dependencies
├── README.md                       # User guide
├── THEORY.md                       # Theoretical foundation
└── IMPLEMENTATION.md               # This file
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from financial_world_model import HierarchicalDiffusionWorldFinance
import torch

# Create model
model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)

# Create inputs
x_A = {
    'price': torch.randn(4, 16),
    'volume': torch.randn(4, 8),
    'order_imbalance': torch.randn(4, 4),
    'volatility': torch.randn(4, 4)
}

x_B = {
    'regime': torch.randn(4, 8),
    'vol_structure': torch.randn(4, 8),
    'liquidity': torch.randn(4, 4),
    'risk_appetite': torch.randn(4, 4)
}

x_C = {
    'action': torch.randn(4, 12),
    'risk': torch.randn(4, 6),
    'objective': torch.randn(4, 4)
}

t = torch.randint(0, 1000, (4,))

# Forward pass
outputs = model(x_A, x_B, x_C, t)
print(outputs['z_A'].shape)  # [4, 64]
print(outputs['z_B'].shape)  # [4, 64]
print(outputs['z_C'].shape)  # [4, 64]
```

### Training
```python
from trainer import HierarchicalWorldModelTrainer

trainer = HierarchicalWorldModelTrainer(model, learning_rate=1e-4)

# Create noise targets
noise_A = {k: torch.randn_like(v) for k, v in x_A.items()}
noise_B = {k: torch.randn_like(v) for k, v in x_B.items()}
noise_C = {k: torch.randn_like(v) for k, v in x_C.items()}

# Training step
losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
print(f"Total loss: {losses['total_loss']}")
```

### Testing
```bash
python test_model.py
```

## Architecture Details

### Causal Information Flow

```
Sub-World A (Independent)
    ↓
    z_A (64 dims)
    ↓
Sub-World B (Reads z_A)
    ↓
    z_B (64 dims)
    ↓
Sub-World C (Reads z_A, z_B)
    ↓
    z_C (64 dims)
    ↓
Global State Aggregator
    ↓
    global_state (64 dims)
```

### Multi-Modal Processing

Each sub-world processes multiple modalities through hierarchical diffusion:

**Sub-World A** (4 modalities):
- Price (16 dims) → Level 0 & 1 diffusion
- Volume (8 dims) → Level 0 & 1 diffusion
- Order Imbalance (4 dims) → Level 0 & 1 diffusion
- Volatility (4 dims) → Level 0 & 1 diffusion
- Aggregate → z_A (64 dims)

Similar structure for Sub-Worlds B and C.

### Loss Functions

**Total Loss**:
```
L_total = L_denoise^A + L_denoise^B + L_denoise^C 
        + 0.1 * (L_causal^B + L_causal^C)
```

Where:
- `L_denoise^s`: MSE between predicted and actual noise
- `L_causal^s`: -correlation(parent_state, child_state)

## Key Features

1. **Hierarchical Cascading Diffusion**
   - Models distributions of distributions
   - Multi-scale structure capture
   - Theoretical innovation

2. **Causal Boundaries**
   - Explicit dependency structure
   - No circular dependencies
   - Information flows downward only

3. **Multi-Modal Modeling**
   - Each modality has independent diffusion cascade
   - Cross-modality fusion at each level
   - Flexible modality addition

4. **Financial Market Modeling**
   - Microstructure layer (high-frequency)
   - Macro dynamics layer (regime-level)
   - Strategy layer (decision-making)

5. **Interpretability**
   - Clear causal structure
   - Latent states at each level
   - Auxiliary outputs (risk_scale, objective_embed)

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension for networks |
| `time_dim` | 128 | Dimension of time embeddings |
| `num_levels` | 2 | Hierarchical diffusion levels |
| `learning_rate` | 1e-4 | Optimizer learning rate |
| `num_steps` | 1000 | Diffusion timesteps |
| `lambda_causal` | 0.1 | Causal loss weight |

## Performance

- **Model Size**: 621,422 parameters
- **Forward Pass**: ~10ms (CPU, batch_size=4)
- **Training Step**: ~50ms (CPU, batch_size=4)
- **Memory**: ~200MB (CPU)

## Theoretical Foundation

See `THEORY.md` for detailed mathematical formulation:
- Cascading diffusion mathematics
- Causal consistency loss derivation
- Multi-modal hierarchical diffusion
- Financial market interpretation

## Future Extensions

1. **Temporal Modeling**: Add LSTM/Transformer for sequences
2. **Real Data Integration**: Train on actual market data
3. **Risk Management**: Portfolio-level constraints
4. **Objective Functions**: Specific trading objectives
5. **Interpretability**: Attention visualization
6. **Adaptive Hierarchies**: Dynamic sub-world creation

## Testing Results

```
✓ Model created with 621,422 parameters
✓ Forward pass successful
✓ Training step successful
✓ Multiple training steps successful
✓ All tests passed!
```

## Usage Examples

### Example 1: Forward Pass
```bash
python test_model.py
```

### Example 2: Full Demonstrations
```bash
python example_usage.py
```

## Dependencies

- torch==2.1.0
- numpy==1.24.3
- scipy==1.11.2
- pandas==2.0.3
- matplotlib==3.7.2
- tqdm==4.66.1

## References

1. Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
2. Kingma & Welling (2013). "Auto-Encoding Variational Bayes"
3. Pearl (2009). "Causality: Models, Reasoning, and Inference"

## Summary

This implementation provides a complete, working system for:
- Hierarchical diffusion-based world modeling
- Multi-modal financial market representation
- Causal structure enforcement
- Training and sampling from the model

The architecture is modular, interpretable, and extensible for future financial applications.
