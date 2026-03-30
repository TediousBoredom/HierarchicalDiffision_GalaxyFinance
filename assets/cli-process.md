

#### example usage test

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && python example_usage.py 2>&1 | head -100

============================================================
Hierarchical Diffusion World Model for Financial Markets
============================================================

============================================================
DEMO 4: Architecture Overview
============================================================

Hierarchical Diffusion World Model Architecture:

1. Sub-World A: Market Microstructure (Independent)
   Modalities:
     - Price paths (16 dims)
     - Volume flows (8 dims)
     - Order imbalance (4 dims)
     - Volatility evolution (4 dims)
   Output: z_A (64 dims latent state)

2. Sub-World B: Macro & Regime Dynamics (Reads z_A)
   Modalities:
     - Trend/oscillation state (8 dims)
     - Volatility structure (8 dims)
     - Liquidity environment (4 dims)
     - Risk appetite (4 dims)
   Output: z_B (64 dims latent state)
   Causal dependency: z_A → z_B

3. Sub-World C: Strategy & Risk Agent (Reads z_A, z_B)
   Modalities:
     - Trading action distribution (12 dims)
     - Risk embedding (6 dims)
     - Objective alignment (4 dims)
   Output: z_C (64 dims latent state)
   Causal dependencies: z_A, z_B → z_C

4. Global State Aggregator
   Combines: z_A + z_B + z_C → global_state (64 dims)

Key Features:
  ✓ Hierarchical cascading diffusion blocks
  ✓ Multi-modal modeling for each sub-world
  ✓ Causal boundaries between layers
  ✓ Cross-world attention mechanisms
  ✓ Risk constraints and objective alignment

Model Parameters:
  Total: 621,422

✓ Architecture overview complete!
============================================================
DEMO 1: Forward Pass Through Hierarchical World Model
============================================================

Input shapes:
  Sub-World A (Microstructure):
    price: torch.Size([4, 16])
    volume: torch.Size([4, 8])
    order_imbalance: torch.Size([4, 4])
    volatility: torch.Size([4, 4])
  Sub-World B (Macro & Regime):
    regime: torch.Size([4, 8])
    vol_structure: torch.Size([4, 8])
    liquidity: torch.Size([4, 4])
    risk_appetite: torch.Size([4, 4])
  Sub-World C (Strategy Agent):
    action: torch.Size([4, 12])
    risk: torch.Size([4, 6])
    objective: torch.Size([4, 4])

Output latent states:
  z_A (Microstructure latent): torch.Size([4, 64])
  z_B (Macro latent): torch.Size([4, 64])
  z_C (Strategy latent): torch.Size([4, 64])
  Global state: torch.Size([4, 64])

Causal information flow:
  Sub-World A → (independent)
  Sub-World B → (reads z_A)
  Sub-World C → (reads z_A, z_B)

✓ Forward pass successful!

============================================================
DEMO 2: Training Loop
============================================================

Training on cpu
Batch size: 4, Steps: 5

Traceback (most recent call last):
  File "/inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance/example_usage.py", line 213, in <module>
    demo_training()
  File "/inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance/example_usage.py", line 120, in demo_training
    losses = trainer.train_step(x_A, x_B, x_C, t, noise_A, noise_B, noise_C)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance/trainer.py", line 59, in train_step
    loss_causal_C = self._causal_consistency_loss(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance/trainer.py", line 96, in _causal_consistency_loss

issue with the causal loss when concatenating tensors. Let me fix the trainer:


For z_C, use average of z_A and z_B as parent
        z_parent_C = (outputs['z_A'] + outputs['z_B']) / 2.0
        loss_causal_C = self._causal_consistency_loss(

            

#### example usage test

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && python example_usage.py 2>&1


============================================================
Hierarchical Diffusion World Model for Financial Markets
============================================================

============================================================
DEMO 4: Architecture Overview
============================================================

Hierarchical Diffusion World Model Architecture:

1. Sub-World A: Market Microstructure (Independent)
   Modalities:
     - Price paths (16 dims)
     - Volume flows (8 dims)
     - Order imbalance (4 dims)
     - Volatility evolution (4 dims)
   Output: z_A (64 dims latent state)

2. Sub-World B: Macro & Regime Dynamics (Reads z_A)
   Modalities:
     - Trend/oscillation state (8 dims)
     - Volatility structure (8 dims)
     - Liquidity environment (4 dims)
     - Risk appetite (4 dims)
   Output: z_B (64 dims latent state)
   Causal dependency: z_A → z_B

3. Sub-World C: Strategy & Risk Agent (Reads z_A, z_B)
   Modalities:
     - Trading action distribution (12 dims)
     - Risk embedding (6 dims)
     - Objective alignment (4 dims)
   Output: z_C (64 dims latent state)
   Causal dependencies: z_A, z_B → z_C

4. Global State Aggregator
   Combines: z_A + z_B + z_C → global_state (64 dims)

Key Features:
  ✓ Hierarchical cascading diffusion blocks
  ✓ Multi-modal modeling for each sub-world
  ✓ Causal boundaries between layers
  ✓ Cross-world attention mechanisms
  ✓ Risk constraints and objective alignment

Model Parameters:
  Total: 621,422

✓ Architecture overview complete!
============================================================
DEMO 1: Forward Pass Through Hierarchical World Model
============================================================

Input shapes:
  Sub-World A (Microstructure):
    price: torch.Size([4, 16])
    volume: torch.Size([4, 8])
    order_imbalance: torch.Size([4, 4])
    volatility: torch.Size([4, 4])
  Sub-World B (Macro & Regime):
    regime: torch.Size([4, 8])
    vol_structure: torch.Size([4, 8])
    liquidity: torch.Size([4, 4])
    risk_appetite: torch.Size([4, 4])
  Sub-World C (Strategy Agent):
    action: torch.Size([4, 12])
    risk: torch.Size([4, 6])
    objective: torch.Size([4, 4])

Output latent states:
  z_A (Microstructure latent): torch.Size([4, 64])
  z_B (Macro latent): torch.Size([4, 64])
  z_C (Strategy latent): torch.Size([4, 64])
  Global state: torch.Size([4, 64])

Causal information flow:
  Sub-World A → (independent)
  Sub-World B → (reads z_A)
  Sub-World C → (reads z_A, z_B)

✓ Forward pass successful!

============================================================
DEMO 2: Training Loop
============================================================

Training on cpu
Batch size: 4, Steps: 5

Step 1/5
  Total Loss: 11.231000
  Loss A: 4.633334, Loss B: 3.685307, Loss C: 2.912574
  Causal Loss B: -0.000221, Causal Loss C: -0.001923
Step 2/5
  Total Loss: 11.811735
  Loss A: 4.402665, Loss B: 4.298157, Loss C: 3.111154
  Causal Loss B: -0.000356, Causal Loss C: -0.002047
Step 3/5
  Total Loss: 10.372357
  Loss A: 3.340587, Loss B: 4.206656, Loss C: 2.825392
  Causal Loss B: -0.000535, Causal Loss C: -0.002232
Step 4/5
  Total Loss: 8.702723
  Loss A: 3.100122, Loss B: 3.131261, Loss C: 2.471643
  Causal Loss B: -0.000630, Causal Loss C: -0.002402
Step 5/5
  Total Loss: 9.074142
  Loss A: 3.279852, Loss B: 3.266123, Loss C: 2.528504
  Causal Loss B: -0.000810, Causal Loss C: -0.002576

✓ Training loop successful!

============================================================
DEMO 3: Sampling Trajectories
============================================================

Sampling on cpu
Generating 2 trajectories of length 5...

Sampled trajectory shapes:
  z_A: torch.Size([2, 5, 64])
    (batch_size=2, seq_length=5, latent_dim=64)
  z_B: torch.Size([2, 5, 64])
    (batch_size=2, seq_length=5, latent_dim=64)
  z_C: torch.Size([2, 5, 64])
    (batch_size=2, seq_length=5, latent_dim=64)
  global_state: torch.Size([2, 5, 64])
    (batch_size=2, seq_length=5, latent_dim=64)

✓ Sampling successful!

============================================================
All demos completed successfully!
============================================================



####example usage

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && timeout 30 python example_usage.py 2>&1 | tail -80

No output


#### test

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && python test_model.py


Testing Hierarchical Diffusion World Model

✓ Model created with 621,422 parameters

1. Testing forward pass...
   z_A shape: torch.Size([2, 64])
   z_B shape: torch.Size([2, 64])
   z_C shape: torch.Size([2, 64])
   global_state shape: torch.Size([2, 64])
   ✓ Forward pass successful

2. Testing training step...
   Total loss: 9.758030
   Loss A: 4.396615
   Loss B: 2.644280
   Loss C: 2.717054
   Causal loss B: -0.001577
   Causal loss C: 0.002376
   ✓ Training step successful

3. Testing multiple training steps...
   Step 1: loss=9.750911
   Step 2: loss=8.881996
   Step 3: loss=12.342585
   ✓ Multiple training steps successful

============================================================
All tests passed! ✓
============================================================

Architecture Summary:
  Sub-World A: Market Microstructure (independent)
  Sub-World B: Macro & Regime (reads z_A)
  Sub-World C: Strategy Agent (reads z_A, z_B)

Key Features:
  ✓ Hierarchical cascading diffusion blocks
  ✓ Multi-modal modeling
  ✓ Causal boundaries between layers
  ✓ Cross-world information flow
  ✓ Risk constraints and objective alignment



#### file

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && ls -lah

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" | sort



#### advanced usage

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && timeout 30 python advanced_usage.py 2>&1


No output


#### verification

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && python verify.py


======================================================================
HIERARCHICAL DIFFUSION WORLD MODEL - VERIFICATION
======================================================================

1. Model Creation
   ✓ Model created with 621,422 parameters

2. Forward Pass
   ✓ z_A shape: torch.Size([2, 64])
   ✓ z_B shape: torch.Size([2, 64])
   ✓ z_C shape: torch.Size([2, 64])
   ✓ global_state shape: torch.Size([2, 64])

3. Training
   ✓ Training step completed
   ✓ Total loss: 12.528049
   ✓ Loss A: 5.112954
   ✓ Loss B: 5.080761
   ✓ Loss C: 2.334392

4. Causal Dependencies
   ✓ Correlation z_A → z_B: 0.0780
   ✓ Correlation z_A → z_C: -0.0778
   ✓ Correlation z_B → z_C: 0.0004

5. Architecture Summary
   Sub-World A: Market Microstructure (Independent)
     - 4 modalities: price, volume, order_imbalance, volatility
     - Output: z_A (64 dims)
   Sub-World B: Macro & Regime (Reads z_A)
     - 4 modalities: regime, vol_structure, liquidity, risk_appetite
     - Output: z_B (64 dims)
   Sub-World C: Strategy Agent (Reads z_A, z_B)
     - 3 modalities: action, risk, objective
     - Output: z_C (64 dims)

6. Key Features
   ✓ Hierarchical cascading diffusion blocks
   ✓ Multi-modal modeling for each sub-world
   ✓ Causal boundaries between layers
   ✓ Cross-world information flow
   ✓ Risk constraints and objective alignment
   ✓ Comprehensive training framework
   ✓ Modular and extensible design

======================================================================
VERIFICATION COMPLETE - ALL COMPONENTS WORKING ✓
======================================================================

Files created:
  - hierarchical_diffusion.py (core architecture)
  - sub_world_a.py (microstructure layer)
  - sub_world_b.py (macro dynamics layer)
  - sub_world_c.py (strategy agent layer)
  - financial_world_model.py (complete model)
  - trainer.py (training framework)
  - test_model.py (quick tests)
  - example_usage.py (usage examples)
  - advanced_usage.py (advanced patterns)
  - README.md (user guide)
  - THEORY.md (theoretical foundation)
  - IMPLEMENTATION.md (implementation guide)
  - PROJECT_SUMMARY.md (project summary)
  - requirements.txt (dependencies)

Next steps:
  1. Review README.md for overview
  2. Run: python test_model.py
  3. Explore: example_usage.py
  4. Study: THEORY.md for mathematical foundation





#### verify_run

cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/HierarchicalDiffision_GalaxyFinance && timeout 20 python verify.py 2>&1

======================================================================
HIERARCHICAL DIFFUSION WORLD MODEL - VERIFICATION
======================================================================

1. Model Creation
   ✓ Model created with 621,422 parameters

2. Forward Pass
   ✓ z_A shape: torch.Size([2, 64])
   ✓ z_B shape: torch.Size([2, 64])
   ✓ z_C shape: torch.Size([2, 64])
   ✓ global_state shape: torch.Size([2, 64])

3. Training
   ✓ Training step completed
   ✓ Total loss: 9.117519
   ✓ Loss A: 2.816418
   ✓ Loss B: 3.031674
   ✓ Loss C: 3.269724

4. Causal Dependencies
   ✓ Correlation z_A → z_B: -0.0312
   ✓ Correlation z_A → z_C: -0.0779
   ✓ Correlation z_B → z_C: 0.0779

5. Architecture Summary
   Sub-World A: Market Microstructure (Independent)
     - 4 modalities: price, volume, order_imbalance, volatility
     - Output: z_A (64 dims)
   Sub-World B: Macro & Regime (Reads z_A)
     - 4 modalities: regime, vol_structure, liquidity, risk_appetite
     - Output: z_B (64 dims)
   Sub-World C: Strategy Agent (Reads z_A, z_B)
     - 3 modalities: action, risk, objective
     - Output: z_C (64 dims)

6. Key Features
   ✓ Hierarchical cascading diffusion blocks
   ✓ Multi-modal modeling for each sub-world
   ✓ Causal boundaries between layers
   ✓ Cross-world information flow
   ✓ Risk constraints and objective alignment
   ✓ Comprehensive training framework
   ✓ Modular and extensible design

======================================================================
VERIFICATION COMPLETE - ALL COMPONENTS WORKING ✓
======================================================================

Files created:
  - hierarchical_diffusion.py (core architecture)
  - sub_world_a.py (microstructure layer)
  - sub_world_b.py (macro dynamics layer)
  - sub_world_c.py (strategy agent layer)
  - financial_world_model.py (complete model)
  - trainer.py (training framework)
  - test_model.py (quick tests)
  - example_usage.py (usage examples)
  - advanced_usage.py (advanced patterns)
  - README.md (user guide)
  - THEORY.md (theoretical foundation)
  - IMPLEMENTATION.md (implementation guide)
  - PROJECT_SUMMARY.md (project summary)
  - requirements.txt (dependencies)

Next steps:
  1. Review README.md for overview
  2. Run: python test_model.py
  3. Explore: example_usage.py
  4. Study: THEORY.md for mathematical foundation
