# Hierarchical Diffusion World Model for Financial Markets

## Overview

This project implements a novel **Hierarchical Diffusion Architecture** for modeling financial markets as a multi-modal causal world model with three hierarchical sub-worlds. The architecture innovates on traditional diffusion models by implementing **cascading diffusion blocks** that model distributions of distributions.

### Core Innovation: Cascading Diffusion

The key theoretical contribution is modeling:
- **Level 0**: p(x | noise) - base data distribution
- **Level 1**: p(z₁ | noise) - distribution of latent codes  
- **Level 2+**: p(z_l | noise) - distribution of distributions

This enables the model to capture multi-scale hierarchical structure in financial markets.

## Architecture

### Three-Layer Hierarchical Structure

```
Global Diffusion World Model
│
├── Sub-World A: Market Microstructure (Independent)
│   ├── Price paths
│   ├── Volume flows
│   ├── Order imbalance
│   └── Volatility evolution
│   └── Output: z_A (latent state)
│
├── Sub-World B: Macro & Regime Dynamics (Reads z_A)
│   ├── Trend/oscillation states
│   ├── Volatility structure
│   ├── Liquidity environment
│   └── Risk appetite
│   └── Output: z_B (latent state)
│
└── Sub-World C: Strategy & Risk Agent (Reads z_A, z_B)
    ├── Trading action distribution
    ├── Risk embedding
    ├── Objective alignment
    └── Output: z_C (latent state)
```

### Causal Boundaries

- **Sub-World A**: Independent, models microstructure
- **Sub-World B**: Reads z_A (causal dependency), models macro dynamics
- **Sub-World C**: Reads z_A and z_B (causal dependencies), generates trading actions

This ensures **causal locality** - each layer only depends on lower layers, preventing circular dependencies.

## Key Components

### 1. HierarchicalDiffusionBlock
- Implements cascading diffusion through multiple levels
- Each level models the distribution of the previous level's latent space
- Includes positional encoding for timesteps
- Variance schedule for noise injection

### 2. MultiScaleHierarchicalDiffusion
- Handles multiple modalities (price, volume, etc.)
- Each modality has its own hierarchical diffusion cascade
- Cross-modality attention for information flow

### 3. Sub-World Modules
- **SubWorldA_MicroStructure**: Independent microstructure modeling
- **SubWorldB_MacroRegime**: Macro dynamics with z_A dependency
- **SubWorldC_StrategyAgent**: Action generation with z_A, z_B dependencies

### 4. Training Framework
- **DiffusionLoss**: MSE loss for noise prediction
- **HierarchicalWorldModelTrainer**: Training loop with causal consistency losses
- **DiffusionSampler**: Trajectory generation via reverse diffusion

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Forward Pass

```python
from financial_world_model import HierarchicalDiffusionWorldFinance
import torch

model = HierarchicalDiffusionWorldFinance(hidden_dim=64, time_dim=128)

# Create inputs for each sub-world
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

# Access latent states
z_A = outputs['z_A']  # Microstructure latent
z_B = outputs['z_B']  # Macro latent
z_C = outputs['z_C']  # Strategy latent
global_state = outputs['global_state']  # Aggregated state
```

### Training

```python
from trainer import HierarchicalWorldModelTrainer

trainer = HierarchicalWorldModelTrainer(model, learning_rate=1e-4)

# Training step
losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
print(f"Total loss: {losses['total_loss']}")
```

### Sampling

```python
from trainer import DiffusionSampler

sampler = DiffusionSampler(model, num_steps=1000)

# Generate trajectories
trajectories = sampler.sample(batch_size=4, seq_length=10)
# trajectories['z_A']: [batch_size, seq_length, latent_dim]
# trajectories['z_B']: [batch_size, seq_length, latent_dim]
# trajectories['z_C']: [batch_size, seq_length, latent_dim]
```

### Run Examples

```bash
python example_usage.py
```

This runs 4 demonstrations:
1. Forward pass through the model
2. Training loop
3. Trajectory sampling
4. Architecture overview

## File Structure

```
HierarchicalDiffision_GalaxyFinance/
├── hierarchical_diffusion.py      # Core hierarchical diffusion blocks
├── sub_world_a.py                 # Market microstructure sub-world
├── sub_world_b.py                 # Macro & regime sub-world
├── sub_world_c.py                 # Strategy & risk agent sub-world
├── financial_world_model.py        # Complete world model
├── trainer.py                      # Training and sampling
├── example_usage.py                # Usage examples
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Theoretical Foundation

### Cascading Diffusion

Traditional diffusion models learn: p(x_t | x_{t-1})

Hierarchical diffusion extends this to:
- Level 0: p(x | noise) - data distribution
- Level 1: p(z₁ | noise) - latent distribution
- Level 2: p(z₂ | noise) - meta-distribution

This creates a **distribution of distributions**, enabling multi-scale modeling.

### Causal Consistency Loss

The model enforces causal dependencies through:
```
loss_causal = -correlation(parent_state, child_state)
```

This ensures child latent states are predictable from parent states, maintaining causal locality.

### Multi-Modal Modeling

Each sub-world handles multiple modalities:
- Sub-World A: 4 modalities (price, volume, order, volatility)
- Sub-World B: 4 modalities (regime, vol_structure, liquidity, risk)
- Sub-World C: 3 modalities (action, risk, objective)

Each modality has its own hierarchical diffusion cascade.

## Financial Market Modeling

### Sub-World A: Microstructure
Models high-frequency market dynamics:
- Price paths: intraday price movements
- Volume flows: trading volume patterns
- Order imbalance: buy/sell pressure
- Volatility: price volatility metrics

### Sub-World B: Macro & Regime
Models market-wide dynamics:
- Regime states: trending vs. ranging markets
- Volatility structure: term structure of volatility
- Liquidity: market liquidity conditions
- Risk appetite: investor risk preferences

### Sub-World C: Strategy Agent
Generates trading decisions:
- Action distribution: probability of different actions
- Risk embedding: risk constraints
- Objective alignment: alignment with trading objectives

## Loss Functions

1. **Denoising Loss**: MSE between predicted and actual noise
2. **Causal Consistency Loss**: Ensures child states depend on parent states
3. **Total Loss**: Weighted combination of all losses

## Hyperparameters

- `hidden_dim`: Hidden dimension for networks (default: 64)
- `time_dim`: Dimension of time embeddings (default: 128)
- `num_levels`: Number of hierarchical diffusion levels (default: 2)
- `learning_rate`: Optimizer learning rate (default: 1e-4)
- `num_steps`: Number of diffusion steps (default: 1000)

## Future Extensions

1. **Temporal Modeling**: Add LSTM/Transformer for sequence modeling
2. **Real Data Integration**: Train on actual financial market data
3. **Risk Constraints**: Implement portfolio-level risk constraints
4. **Objective Functions**: Add specific trading objectives (profit, Sharpe ratio, etc.)
5. **Interpretability**: Add attention visualization and latent space analysis

## References

- Diffusion Models: Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
- Hierarchical Models: Kingma & Welling (2013), "Auto-Encoding Variational Bayes"
- Causal Models: Pearl (2009), "Causality: Models, Reasoning, and Inference"

## License

MIT License

## Contact

For questions or suggestions, please open an issue or contact the development team.
