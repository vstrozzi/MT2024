import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os
import einops
import sympy
import json
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
from utils.initialization_text_set import *

#torch.manual_seed(420)
#np.random.seed(420)

def splice_data_approx(data, text_features, texts, layer, head, seed, dataset, iters, rank, device):
    """
    This function performs a positive least square approximation of the attention head matrix
    using the provided text features.

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        iters: Number of iterations to perform.
        rank: The rank of the approximation matrix (i.e. # of text_features to preserve).
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix.
        results: List of text descriptions with maximum variance.
    """

    # Setup Writer
    writer = SummaryWriter("logs")
    print(f"\nLayer [{layer}], Head: {head}")
    # Define tag prefixes for this layer, head, and seed
    tag_prefix = f"Dataset_{dataset}/Layer_{layer}/Head_{head}/Seed_{seed}"
    
    # Center text and image data
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Project text_features to data lower-rank eigenspace with required rank
    u, s, vh = torch.linalg.svd(data, full_matrices=False)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine rank where cumulative variance exceeds 99% of total variance
    threshold = 0.99
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank]
    s = s[:rank]

    # Prohect features in text space
    text_features = text_features @ vh.T @ vh
    text_features = text_features / torch.linalg.norm(text_features, axis=-1)[:, np.newaxis]

    # Get initialization parameters
    indexes, strength = svd_parameters_init(vh, s, text_features, rank)
    
    # Initialize A with required gradient and with initial range guess
    strength = strength - strength.min()/(strength.max() - strength.min())
    A = torch.empty(data.shape[0], text_features.shape[0], device=device)
    A[:, indexes] = 2*data.max()*(strength.unsqueeze(0)*torch.rand(data.shape[0], indexes.shape[0],  device=device))
    mask = torch.ones(A.shape[1], dtype=bool, device=device)
    mask[indexes] = False  # Set the specified indexes to False
    A[:, mask] = 2*data.max()*torch.min(strength)*torch.rand(data.shape[0], mask.shape[0] - torch.unique(indexes).shape[0],  device=device)
    A_ = A.clone().detach().requires_grad_(True)

    # Set up optimizer and parameters
    optimizer = torch.optim.Adam([A_], lr=0.001)
    epochs_main = 7000
    
    # Initial ratio bewteen regularization and rmse loss 
    lbd_l1 = 1
    ratio = 1

    ## First part: main optimization loop
    patience = 500  # Number of epochs to log something

    # Initialize variables for early stopping
    prev_cum_sum = None
    prev_indexes = torch.tensor([x for x in range(iters)], device=device)
    prev_relative_strength = None
    stabilization_window = 2000  # Number of iterations to check stability
    cum_sum_stable_count = 0  # Counter for indexes change stability
    relative_strength_stable_count = 0  # Counter for relative strength stability
    stabilization_threshold_cum = int(min(iters, rank)) + 1 # Percentage
    stabilization_threshold_strength = 0.01    
    tolerance = 1e-9

    # Training loop with early stopping
    for epoch in range(epochs_main):
        
        optimizer.zero_grad()  # Clear gradients from previous step

        # Clip paramteres to keep them positive
        A = torch.nn.functional.relu(A_)

        # Compute the product A @ text_features using only stronger "iters" text with highest std across data
        text_features_mean = A.mean(axis=0)
        indexes = torch.argsort(text_features_mean, descending=True)[:iters]
        pred = A[:, indexes] @ text_features[indexes, :]

        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred-data)**2))
        # Regularization L1 on row *used* for predictions (i.e. sparse row i.e. fewer text embeddings)
        # and L_inf on *used* for predictions columns
        loss_l1 = ratio * lbd_l1 * (torch.norm(A[:, indexes], p=1, dim=1).mean() + \
                            torch.norm(A[:, indexes], p=float('inf'), dim=0).mean())

        loss = loss_l1 + loss_rmse

        # Use a lbd_1 of 1:1 of loss functions
        if epoch == 0:
            lbd_l1 = ratio * lbd_l1 * loss_rmse.detach().clone()/loss_l1.detach().clone()
            epoch += 1
            continue
        
        # Backpropagation
        loss.backward()
        
        # Update A using the optimizer
        optimizer.step()

        # Compute metrics for early stopping
        text_str = text_features_mean[indexes]
        tot_str = torch.sum(text_str) + 1e-9
        all_str = torch.sum(text_features_mean) + 1e-9

        relative_strength = 100 * tot_str / all_str

        # Calculate the cumulative sum
        cum_sum = (prev_indexes[:stabilization_threshold_cum] != indexes[:stabilization_threshold_cum]).sum()

        # Log to TensorBoard every patience epochs
        if epoch % patience == 0:
            # Log the loss to TensorBoard
            writer.add_scalar(f"{tag_prefix}/Loss/RMSE", loss_rmse, epoch)
            writer.add_scalar(f"{tag_prefix}/Loss/L1", loss_l1, epoch)
            writer.add_scalar(f"{tag_prefix}/Loss/loss", loss, epoch)
 
            # Log additional metadata (e.g., loss or total strength)
            writer.add_scalar(f"{tag_prefix}/relative_strength", relative_strength, epoch)

            writer.add_scalar(f"{tag_prefix}/indexes_cum_sum", cum_sum, epoch)

        # Check stability for `relative_strength`
        if prev_relative_strength is not None:
            if relative_strength > 99.9:
                relative_strength_stable_count += 1
            else:
                relative_strength_stable_count = 0  # Reset if not stable
        # Check stability for `cum_sum`
        if prev_cum_sum is not None:
            cum_sum_change = abs(cum_sum - prev_cum_sum)
            if cum_sum_change < 1:
                cum_sum_stable_count += 1
            else:
                cum_sum_stable_count = 0  # Reset if not stable

        # Update previous values
        prev_relative_strength = relative_strength
        prev_cum_sum = cum_sum
        prev_indexes = indexes

        
    # Log text features
    text_str = text_features_mean[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_mean)   
    results = [
        {
            "text": texts[idx],
            "strength_abs": text_str[i].item(),
            "strength_rel": (100 * text_str[i] / tot_str).item(),
        }
        for i, idx in enumerate(indexes)
    ]
    # Generate Markdown-formatted table
    markdown_table = "| Text         | Absolute Strength | Relative Strength (%) |\n"
    markdown_table += "|--------------|-------------------|------------------------|\n"
    for result in results:
        markdown_table += f"| {result['text']} | {result['strength_abs']:.4f}         | {result['strength_rel']:.2f}                   |\n"
    # Log LaTeX table as text (enable slider with epoch step)
    writer.add_text(f"{tag_prefix}/Top-K Strengths for Epoch", markdown_table, epoch)   


    # Take columns of A with highest mean (i.e. more active columns -> more active text embedding)
    text_features_mean = A.mean(axis=0)
    indexes = torch.argsort(text_features_mean, descending=True)[:iters]
    A = A[:, indexes].detach().clone().requires_grad_(True)
    text_features = text_features[indexes, :].detach().clone().requires_grad_(True)

    ## Second part: finetune over rmse loss
    # Training loop with early stopping
    patience_counter = 0
    patience = 100  # Number of epochs to wait for improvement
    min_delta = 1e-9  # Minimum improvement in loss to be considered
    A_ = A.requires_grad_(True) 
    A_.data.clamp_(0)

    optimizer = torch.optim.Adam([A], lr=0.001)  # Recreate the optimizer
    epochs = 500
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Clear gradients from previous step
        optimizer.zero_grad()  

        # Clip paramteres to keep them positive
        A = torch.nn.functional.relu(A_)

        # Make prediction
        pred = A @ text_features
        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred-data)**2))
        loss = loss_rmse
        # Backpropagation
        loss.backward()
        # Update A using the optimizer
        optimizer.step()

        # Log values
        writer.add_scalar(f"{tag_prefix}/Loss/RMSE", loss_rmse, epochs_main + epoch) 
        writer.add_histogram(f"{tag_prefix}/indexes", indexes, epochs_main + epoch)

        # Early stopping logic
        if loss < best_loss - min_delta:
            best_loss = loss_rmse
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1  # Increment if no improvement

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} with best loss_rmse: {best_loss.item():.6f}")
            break

    # Log second time  
    text_str = text_features_mean[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_mean)   
    results = [
        {
            "text": texts[idx],
            "strength_abs": text_str[i].item(),
            "strength_rel": (100 * text_str[i] / tot_str).item(),
        }
        for i, idx in enumerate(indexes)
    ]
    # Generate Markdown-formatted table
    markdown_table = "| Text         | Absolute Strength | Relative Strength (%) |\n"
    markdown_table += "|--------------|-------------------|------------------------|\n"
    for result in results:
        markdown_table += f"| {result['text']} | {result['strength_abs']:.4f}         | {result['strength_rel']:.2f}                   |\n"
    # Log LaTeX table as text (enable slider with epoch step)
    writer.add_text(f"{tag_prefix}/Top-K Strengths for Epoch", markdown_table, epochs_main + epoch)   

    writer.close()

    # Retrieve corresponding text
    text_str = text_features_mean[indexes].cpu()
    tot_str = torch.sum(text_str).cpu() # Total strength of text embeddings on that
    results = [{"text": texts[idx], "strength_abs": text_str[i].item(), "strength_rel": (100 * text_str[i] / tot_str).item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Compute reconstruction using only our selected 
    A = A.clamp(0, None)
    reconstruct = A.detach().cpu().numpy() @ text_features.detach().cpu().numpy()
    
    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object

    