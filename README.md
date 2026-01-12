# Compositional-Visual-Generation-Project
Compositional Visual Generation via LoRA-Enhanced Stable Diffusion with Depth Conditioning

Group Course Project

Objective: Fine-tune the Stable Diffusion (SD)
model for the LAION SG Dataset [Li et al., 2024] to 
improve text-to-image generation from complex 
compositional prompts

[Full paper and Project Explanation Here](./paper.pdf)


INTRODUCTION

➢ Task: Fine-tune Stable Diffusion v2.1 to handle 
complex compositional prompts. 

Model must correctly capture:
- Attribute binding
- Object relationships
- Multiple object relationship
- Spatial reasoning
  
➢ Dataset: LAION-SG dataset contains 20 million 
image-text pairs from LAION-5B augmented with 
scene graph annotations.

Each image includes: 
- Object names
- Attributes (color, size, material)
- Spatial relationships ("left of," "above," "inside")


PROBLEM

Text-to-image diffusion models 
may fail on compositional prompts 
with multiple objects, attributes, 
and spatial relations

➢ wrong attribute binding
➢ missing objects
➢ bad spatial layout

Ex: The red book was on top of the 
yellow bookshelf


METHODS

To explore the impact of fine-tuning techniques of 
different modules, the model was trained with the 
following configs:

1. Frozen text encoder + LoRA U-Net attention layers + fully trained Depth Predictor
2. Fully fine-tuned text encoder + LoRA U-Net 
attention layers + fully trained Depth Predictor


➢ Convert scene graph annotations from LAION-SG 
into compositional text prompts
- "red tree (large)" + "wooden bench (small)" + 
"next to" → prompt "a large red tree next to a 
small wooden bench"

➢ Introduce Depth Map Predictor module that 
generates depth representations directly from text 
embeddings

 - Transformer-based decoder architecture with 
 learnable spatial embeddings for 64×64 grid as 
 queries

 - 4 cross-attention transformer decoder layers 
 (self-attention, cross-attention with CLIP text 
 tokens, feed-forward networks)

 - Outputs reshaped to 64×64×256, then processed 
 through convolutional layers with sigmoid 
 activation to produce single-channel normalized 
 depth map.

➢ Modify SD U-Net to accept 5-channel input (4 
standard latent channels + 1 depth channel)



EVALUATION 

We evaluate using metrics proposed 
by T2I-CompBench:

- Attribute Binding Accuracy: Measures correct 
attribute-object associations

- Object Relationship Score: Evaluates spatial and 
relational accuracy between objects

- Generative Numeracy: Assesses correct 
generation of specified object counts

- Complex Composition Score: Evaluates 
performance on multi-element compositional 
prompts



