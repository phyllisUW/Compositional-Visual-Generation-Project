# LOAD TRAINED TRANSFORMER DEPTH MODEL (SINGLE-SCALE)
import torch
import torch.nn as nn
from pathlib import Path

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Define the model architecture (must match training)
class SimpleTransformerDepthPredictor(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, num_layers=4, num_heads=4):
        super().__init__()
        
        # Learnable spatial queries for 64x64 positions
        self.spatial_queries = nn.Parameter(torch.randn(4096, hidden_dim))
        
        # Project text embeddings to match query dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, text_embeddings):
        # text_embeddings: [B, 77, 768]
        B = text_embeddings.shape[0]
        
        # Project text
        text_feat = self.text_proj(text_embeddings)  # [B, 77, 256]
        
        # Prepare queries
        queries = self.spatial_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 4096, 256]
        
        # Cross-attention
        output = self.transformer(queries, text_feat)  # [B, 4096, 256]
        
        # Reshape to spatial
        output = output.transpose(1, 2).reshape(B, 256, 64, 64)  # [B, 256, 64, 64]
        
        # Generate depth map
        depth = self.output_conv(output)  # [B, 1, 64, 64]
        
        return depth

# Initialize model
model = SimpleTransformerDepthPredictor(
    text_dim=768,
    hidden_dim=256,
    num_layers=4,
    num_heads=4
)
model = model.to(device)

# Load trained weights
checkpoint_path = 'depth_predictor_checkpoints_transformer_decoder/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Validation loss: {checkpoint['val_loss']:.4f}")