import torch
import timm

# Load the ViT-B-16 model
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Count the total number of parameters
num_params = sum(p.numel() for p in model.parameters())

print(f"Number of parameters in ViT-B-16: {num_params}")
