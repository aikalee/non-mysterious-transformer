## Overview
This project explores the geometry of residual networks through a low-dimensional toy model.
Instead of studying high-dimensional transformer representations (e.g. 768-dimensional hidden states), a 2D residual network is contructed where the representation dynamics can be directly visualized.
Each layer updates the hidden state via a residual update:

​
 
This allows interpretation of deep networks as trajectories in representation space.

## Motivation
Transformer models are often perceived as highly complex systems whose internal behavior is difficult to interpret. This perception may partly arise from the high dimensionality of their representations rather than from fundamentally complicated mechanisms.

A transformer can instead be viewed as a composition of relatively simple models stacked together. Each layer aggregates information from earlier layers and learns progressively higher-level representations. The overall behavior of the model therefore emerges from iterative composition across layers.

To make this process directly observable, the hidden dimension is constrained to two dimensions. This enables direct visualization of how representations evolve through the network, including:
- layer-wise representation trajectories
- the effect of LayerNorm on representation dynamics

## Dataset

We generate a synthetic 2D classification dataset consisting of concentric classes.

The dataset requires nonlinear decision boundaries, making it a suitable testbed for studying representation transformations.

## Model
A simplified residual network:

Each residual block consists of:

## Experiments
Two settings were compared:
- With LayerNorm
- Without LayerNorm
  
For each model, the followings are visualized:
1. Residual vector field

Mapping from input coordinate to residual updates:
[equation]

This shows how the network reshapes the feature space.

2. Representation trajectories
For a sampled point:
[equation]

This reveals how the representation evolves across layers.

## Observations
[something]

## Why a 2D model?

High-dimensional models make representation dynamics difficult to interpret.

This project demonstrates that many residual network behaviours can be understood through low-dimensional geometric intuition.

## Sample Usage
Train the model:
```
python train.py
```
