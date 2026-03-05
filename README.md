## Overview
Transformers can be viewed as systems that iteratively push representations through space.
By constraining the hidden dimension to 2D, we can directly visualize these representation trajectories and residual vector fields.

## Visualization
<img height="300" alt="layer1_with_ln" src="https://github.com/user-attachments/assets/ffe1198d-ab93-446e-b855-cef334b140b3" />
<img height="300" alt="layer1_without_ln" src="https://github.com/user-attachments/assets/0c951559-3a73-4fde-9703-4fa45c678409" />
<br>
<img height="300" alt="layer3_with_ln" src="https://github.com/user-attachments/assets/be185bfe-a8c9-455a-be22-1629e5201417" />
<img height="300" alt="layer3_without_ln" src="https://github.com/user-attachments/assets/6f8d5ad1-1299-416c-ac33-b2c70c69a141" />
<br>
<img height="300" alt="layer6_with_ln" src="https://github.com/user-attachments/assets/87d86130-a1d0-442f-976e-bc2d4c350663" /> 
<img height="300" alt="layer6_without_ln" src="https://github.com/user-attachments/assets/5eabf452-0afe-45bc-b3b5-64cc2f9301d1" />
<br>
<img height="305" alt="trajectory_with_ln" src="https://github.com/user-attachments/assets/b755c3b1-8a26-4ef4-827e-a0a0aa08c814" />
<img height="305" alt="trajectory_without_ln" src="https://github.com/user-attachments/assets/1722ff54-69b8-4185-8bd4-0e5164801714" />

## Interpretation
### Representation Dynamics

A transformer layer can be viewed as repeatedly shifting representations in the hidden space. Each block produces a residual update that moves vectors step by step, forming a trajectory through the representation space.

With LayerNorm, these updates remain relatively balanced, allowing representations to evolve gradually while preserving the overall structure of the three classes. Without LayerNorm, residual updates can become uneven in magnitude, allowing a single update direction to dominate and produce large shifts. This can disrupt the existing structure of the representation space and cause representations from different classes to overlap.

## Motivation
Transformer models are often perceived as highly complex systems whose internal behavior is difficult to interpret. This perception may partly arise from the high dimensionality of their representations rather than from fundamentally complicated mechanisms.

A transformer can instead be viewed as a composition of relatively simple models stacked together. Each layer aggregates information from earlier layers and learns progressively higher-level representations. The overall behavior of the model therefore emerges from iterative composition across layers.

To make this process directly observable, the hidden dimension is constrained to two dimensions. This enables direct visualization of how representations evolve through the network, including:
- layer-wise representation trajectories
- the effect of LayerNorm on representation dynamics

## Dataset

We generate a synthetic 2D classification dataset consisting of 3 concentric classes.

The dataset requires nonlinear decision boundaries, making it a suitable testbed for studying representation transformations.

## Model
A simplified residual network:

<img height="400" alt="model-structure" src="https://github.com/user-attachments/assets/c8c2b7c3-16a9-4e51-a5dc-d316abe17e5a" />

Each residual block consists of:

<img height="400" alt="residual-block" src="https://github.com/user-attachments/assets/63c852b6-c88a-40c1-aa5d-f91390e2bdfc" />

## Experiments
Two settings were compared:
- With LayerNorm
- Without LayerNorm
  
For each model, the followings are visualized:
1. Residual vector field

Mapping from input coordinate to residual updates:

$(x, y) \rightarrow r(x, y)$

This shows how the network reshapes the feature space.

2. Representation trajectories for a sampled point:

$h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow ... \rightarrow h_n$

This reveals how the representation evolves across layers.






## Why a 2D model?

High-dimensional models make representation dynamics difficult to interpret.

This project demonstrates that many residual network behaviours can be understood through low-dimensional geometric intuition.

## Sample Usage
Train the model:
```
python train.py
```
