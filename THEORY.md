"""
Theoretical Foundation: Hierarchical Diffusion Architecture
Novel cascading diffusion framework for multi-scale modeling
"""

# HIERARCHICAL DIFFUSION: THEORETICAL FRAMEWORK
# =============================================

## 1. CORE INNOVATION: CASCADING DIFFUSION

### Traditional Diffusion Model
Standard diffusion models learn the reverse process:
  p(x_{t-1} | x_t) ≈ N(μ_θ(x_t, t), Σ_θ(x_t, t))

Where:
  - x_t: noisy data at timestep t
  - μ_θ: learned mean prediction
  - Σ_θ: learned variance

### Hierarchical Cascading Diffusion (Novel)
We extend this to multiple hierarchical levels:

Level 0 (Base):
  p(x | noise) - models data distribution
  Learns: denoise(noise) → x

Level 1 (Meta):
  p(z₁ | noise) - models latent distribution
  Learns: denoise(noise) → z₁ = encode(x)

Level 2+ (Meta-Meta):
  p(z_l | noise) - models distribution of distributions
  Learns: denoise(noise) → z_l = encode(z_{l-1})

### Mathematical Formulation

For each level l:
  z_l = f_l(z_{l-1}, t, ε)

Where:
  - f_l: denoising network at level l
  - z_{l-1}: latent from previous level
  - t: diffusion timestep
  - ε: noise

The cascade creates:
  noise → z_1 → z_2 → ... → z_L

This models: p(z_L | noise) = ∫ p(z_L | z_{L-1}) p(z_{L-1} | z_{L-2}) ... p(z_1 | noise) dz_1...z_{L-1}

### Key Insight
By modeling distributions of distributions, we capture:
  1. Fine-grained structure at lower levels
  2. Coarse-grained patterns at higher levels
  3. Multi-scale dependencies


## 2. HIERARCHICAL DIFFUSION WORLD MODEL FOR FINANCE

### Architecture Overview

Global Diffusion World Model
│
├── Sub-World A: Microstructure (Level 0)
│   └── Independent modeling of market microstructure
│
├── Sub-World B: Macro Dynamics (Level 1)
│   └── Depends on Sub-World A latent state
│
└── Sub-World C: Strategy Agent (Level 2)
    └── Depends on Sub-World A and B latent states

### Causal Structure

The architecture enforces causal locality:

  z_A → z_B → z_C
  ↓     ↓
  (independent) (reads z_A) (reads z_A, z_B)

This ensures:
  1. No circular dependencies
  2. Information flows downward only
  3. Each layer can be independently trained
  4. Interpretable causal relationships


## 3. SUB-WORLD SPECIFICATIONS

### Sub-World A: Market Microstructure

Modalities:
  - Price paths (16 dims): intraday price movements
  - Volume flows (8 dims): trading volume patterns
  - Order imbalance (4 dims): buy/sell pressure
  - Volatility (4 dims): price volatility metrics

Latent state z_A (64 dims):
  z_A = encode(price, volume, order_imbalance, volatility)

Properties:
  - Independent: no dependencies on other sub-worlds
  - High-frequency: captures tick-level dynamics
  - Foundation: provides input to higher levels


### Sub-World B: Macro & Regime Dynamics

Modalities:
  - Regime states (8 dims): trending vs. ranging
  - Volatility structure (8 dims): vol term structure
  - Liquidity (4 dims): market liquidity conditions
  - Risk appetite (4 dims): investor risk preferences

Latent state z_B (64 dims):
  z_B = encode(regime, vol_structure, liquidity, risk_appetite | z_A)

Causal dependency:
  z_B depends on z_A through cross-world fusion:
  z_B = encoder(features_B ⊕ project(z_A))

Where ⊕ denotes concatenation and project() adapts z_A to feature space.

Properties:
  - Market-wide: captures regime-level dynamics
  - Depends on microstructure: z_A influences z_B
  - Intermediate: bridges micro and strategy levels


### Sub-World C: Strategy & Risk Agent

Modalities:
  - Action distribution (12 dims): trading action probabilities
  - Risk embedding (6 dims): risk constraints
  - Objective alignment (4 dims): strategy objective alignment

Latent state z_C (64 dims):
  z_C = encode(action, risk, objective | z_A, z_B)

Causal dependencies:
  z_C depends on both z_A and z_B:
  z_C = encoder(features_C ⊕ project(z_A) ⊕ project(z_B))

Auxiliary outputs:
  - risk_scale: risk modulation factor
  - objective_embed: objective alignment embedding

Properties:
  - Action generation: produces trading decisions
  - Multi-input: reads both micro and macro states
  - Risk-aware: incorporates risk constraints


## 4. LOSS FUNCTIONS

### Denoising Loss (Per Sub-World)

For each sub-world s ∈ {A, B, C}:
  L_denoise^s = MSE(ŷ_s, y_s)

Where:
  - ŷ_s: predicted noise from diffusion model
  - y_s: actual noise added during forward process

Total denoising loss:
  L_denoise = L_denoise^A + L_denoise^B + L_denoise^C


### Causal Consistency Loss

Ensures child latent states are predictable from parent states:

For Sub-World B:
  L_causal^B = -correlation(z_A, z_B)
  = -E[(z_A - μ_A)(z_B - μ_B)] / (σ_A σ_B)

For Sub-World C:
  L_causal^C = -correlation(z_parent, z_C)
  where z_parent = (z_A + z_B) / 2

Intuition:
  - Negative correlation loss encourages positive correlation
  - Higher correlation = stronger causal dependency
  - Ensures information flows from parent to child


### Total Loss

L_total = L_denoise + λ_causal * (L_causal^B + L_causal^C)

Where λ_causal = 0.1 (hyperparameter controlling causal strength)


## 5. MULTI-MODAL HIERARCHICAL DIFFUSION

### Per-Modality Cascading

Each modality has its own hierarchical diffusion:

For modality m in sub-world s:
  Level 0: p(x_m | noise) - data distribution
  Level 1: p(z_m^1 | noise) - latent distribution
  Level 2: p(z_m^2 | noise) - meta-distribution

### Cross-Modality Fusion

Modalities are fused at each level:
  z_s = aggregate([z_m^l for m in modalities])

Aggregation methods:
  1. Concatenation: z_s = [z_1^l, z_2^l, ..., z_M^l]
  2. Attention: z_s = Attention(z_1^l, z_2^l, ..., z_M^l)
  3. Gating: z_s = Σ_m g_m * z_m^l

Current implementation uses concatenation + learned projection.


## 6. TRAINING PROCEDURE

### Forward Pass
1. Add noise to inputs: x_noisy = √(α_t) * x + √(1-α_t) * ε
2. Pass through Sub-World A: get z_A
3. Pass through Sub-World B with z_A: get z_B
4. Pass through Sub-World C with z_A, z_B: get z_C
5. Compute losses and backpropagate

### Variance Schedule
Linear schedule from β_start=0.0001 to β_end=0.02:
  β_t = β_start + (β_end - β_start) * t / T

Cumulative product:
  ᾱ_t = ∏_{s=1}^t (1 - β_s)

Noise addition:
  x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε


### Optimization
- Optimizer: Adam with learning rate 1e-4
- Scheduler: Cosine annealing over 100 epochs
- Gradient clipping: max norm = 1.0
- Batch size: configurable (default 4)


## 7. SAMPLING / GENERATION

### Reverse Diffusion Process
Starting from pure noise, iteratively denoise:

For t = T, T-1, ..., 1:
  1. Predict noise: ε̂_t = model(x_t, t)
  2. Update: x_{t-1} = (x_t - √(1-ᾱ_t) * ε̂_t) / √(ᾱ_t)
  3. Add noise: x_{t-1} += √(β_t) * z

Result: x_0 ~ p(x)

### Trajectory Generation
Generate sequences by sampling at multiple timesteps:
  trajectory = [z_A^{t_1}, z_A^{t_2}, ..., z_A^{t_K}]

Where t_1 > t_2 > ... > t_K (reverse order)


## 8. FINANCIAL MARKET INTERPRETATION

### Market Microstructure (Sub-World A)
- Captures tick-level dynamics
- Independent of macro conditions
- Foundation for higher-level modeling
- Example: high-frequency trading signals

### Macro Dynamics (Sub-World B)
- Captures regime changes
- Depends on microstructure
- Intermediate abstraction level
- Example: trend identification

### Strategy Agent (Sub-World C)
- Generates trading decisions
- Reads both micro and macro
- Incorporates risk constraints
- Example: portfolio rebalancing


## 9. ADVANTAGES OF HIERARCHICAL APPROACH

1. **Modularity**: Each sub-world can be trained/updated independently
2. **Interpretability**: Clear causal structure and information flow
3. **Scalability**: Easy to add more sub-worlds or modalities
4. **Robustness**: Causal boundaries prevent error propagation
5. **Efficiency**: Hierarchical structure reduces computational complexity
6. **Flexibility**: Can incorporate domain knowledge at each level


## 10. FUTURE EXTENSIONS

1. **Temporal Modeling**: Add LSTM/Transformer for sequence dependencies
2. **Real Data**: Train on actual financial market data
3. **Risk Constraints**: Portfolio-level risk management
4. **Objective Functions**: Specific trading objectives (Sharpe ratio, etc.)
5. **Attention Visualization**: Interpretability of information flow
6. **Adaptive Hierarchies**: Dynamic sub-world creation based on data


## REFERENCES

1. Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
2. Kingma & Welling (2013). "Auto-Encoding Variational Bayes"
3. Pearl (2009). "Causality: Models, Reasoning, and Inference"
4. Vaswani et al. (2017). "Attention Is All You Need"
5. Goodfellow et al. (2016). "Deep Learning" (MIT Press)
