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

torch.manual_seed(420)
np.random.seed(420)
@torch.no_grad()
def text_span(data, text_features, texts, iters, rank, device):
    """
    This function performs iterative removal and reconstruction of the attention head matrix
    using the provided text features.

    Args:
        data: The attention head matrix (data in row).
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        iters: Number of iterations to perform.
        rank: The rank of the approximation matrix (i.e. # of text_features to preserve).
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix.
        results: List of text descriptions with maximum variance.
    """
    results = []
    # Svd of attention head matrix
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    text_features = text_features - mean_values_text
    data = data - mean_values_att
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:rank]
    # return np.matrix(u[:, :80]) * np.diag(s[:80]) * np.matrix(vh[:80, :]) ,results # TASK: this line to test svd
    # Get the projection of text embeddings into head activations matrix space
    text_features = (vh.T @ vh @ text_features.T).T

    data = torch.from_numpy(data).float().to(device)
    reconstruct = np.zeros_like(data)
    text_features = torch.from_numpy(text_features).float().to(device)

    # Reconstruct attention head matrix by using projection on nr. iters max variance texts embeddings
    for i in range(iters):
        # Projects each data point (rows in data) into the feature space defined by the text embeddings.
        # Each row in projection now represents how each attention head activation vector i aligns with each text embedding j (i, j),
        # quantifying the contribution of each text_feature to the data in this iteration.
        projection = data @ text_features.T # Nxd * dxM, cos similarity on each row
        projection_std = projection.std(axis=0).detach().cpu().numpy() 
        # Take top text embedding with max variance for the data matrix
        top_n = np.argmax(projection_std)
        results.append(texts[top_n])
        
        # Rank 1 approximation 
        text_norm = text_features[top_n] @ text_features[top_n].T
        rank_1_approx = (data @ text_features[top_n] / text_norm)[:, np.newaxis]\
                        * text_features[top_n][np.newaxis, :]
        reconstruct += rank_1_approx.detach().cpu().numpy()
        # Remove contribution from data matrix
        data = data - rank_1_approx
        # Remove contribution of text_feature from text embeddings
        text_features = (
            text_features
            - (text_features @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )

    results = [{"text": text} for text in results]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object

def solve_non_negative_least_squares(D, X, num_iter=10000, lr=1e-3):
    """
    Solves for A in D ≈ A * X with A ≥ 0 using PyTorch.

    Parameters:
    D (torch.Tensor): Target matrix of shape (m, n).
    X (torch.Tensor): Input matrix of shape (p, n).
    num_iter (int): Number of optimization iterations.
    lr (float): Learning rate for the optimizer.

    Returns:
    torch.Tensor: Solution matrix A of shape (m, p) with non-negative entries.
    """
    # Ensure X and D are float tensors and on the same device
    X = X.float()
    D = D.float()
    device = X.device
    D = D.to(device)
    
    m, n = D.shape
    p = X.shape[0]
    
    # Initialize A with random values
    A_param = torch.randn(m, p, device=device, requires_grad=True)
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam([A_param], lr=lr)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        # Enforce non-negativity using softplus
        AX = torch.matmul(A_param, X)
        # Compute the loss (Frobenius norm squared)
        loss = torch.norm(D - AX, p='fro')**2
        loss.backward()
        optimizer.step()

        # Clip paramteres to keep them positive
        A_param.data.clamp_(0)

    print(loss)
    return A_param

def svd_data_approx(data, text_features, texts, layer, head, seed, iters, rank, device):
    """
    This function performs SVD-based approximation of the attention head matrix
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

    # Svd of attention head matrix (mean centered)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    text_features = text_features - mean_values_text
    data = data - mean_values_att
    # Subtract the mean from each column
    u, s, vh = np.linalg.svd(data, full_matrices=False)

    # Total sum of singular values
    total_variance = np.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = np.cumsum(s, axis=0)
    # Determine rank where cumulative variance exceeds 99% of total variance
    threshold = 0.99
    rank = np.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank]
    # Divide iters since we want both positive and negative contribution
    iters = rank

    # Get the projection of text embeddings into head activations matrix space
    text_features = text_features @ vh.T @ vh
    text_features = text_features / np.linalg.norm(text_features, axis=-1)[:, np.newaxis]
    # Return the closest text_features in eigen space of data matrix of top iters eigenvector
    simil_matrix = (vh @ text_features.T) # Nxd * dxM, cos similarity on each row
    indexes_max = np.squeeze(simil_matrix.argmax(axis=-1).astype(int))[:iters]
    indexes_min = np.squeeze(simil_matrix.argmin(axis=-1).astype(int))[:iters]

    # Replace duplicates texts
    used_elements_max, used_indexes_max = np.unique(indexes_max, return_index=True)
    idxs_not_unique_max = np.setdiff1d(np.arange(len(indexes_max)), used_indexes_max)
    used_elements_min, used_indexes_min = np.unique(indexes_min, return_index=True)
    idxs_not_unique_min = np.setdiff1d(np.arange(len(indexes_min)), used_indexes_min)
    
    for idx_not_unique in idxs_not_unique_max:
        # Get argsort to find indices of max elements in descending order
        row_argsorted_desc = simil_matrix[idx_not_unique].argsort()[::-1]

        # Find the first argmax that hasn't been used yet
        for index in row_argsorted_desc:
            if index not in used_elements_max:
                indexes_max[idx_not_unique] = index
                used_elements_max = np.append(used_elements_max, index)
                # Replace subsequent duplicates (i.e. give more priority to the first eigenvectors
                # similarity)
                used_elements_max, used_indexes_max = np.unique(indexes_max, return_index=True)
                idxs_not_unique_max = np.setdiff1d(np.arange(len(indexes_max)), used_indexes_max)
                break

    for idx_not_unique in idxs_not_unique_min:
        # Get argsort to find indices of max elements in ascending order
        row_argsorted_asc = simil_matrix[idx_not_unique].argsort()
        # Find the first argmin that hasn't been used yet
        for index in row_argsorted_asc:
            if index not in used_elements_min:
                indexes_min[idx_not_unique] = index
                used_elements_min = np.append(used_elements_min, index)
                # Replace subsequent duplicates (i.e. give more priority to the first eigenvectors
                # similarity)
                used_elements_min, used_indexes_min = np.unique(indexes_min, return_index=True)
                idxs_not_unique_min = np.setdiff1d(np.arange(len(indexes_min)), used_indexes_min)
                break

    # Fix unique texts per eigenvector
    data = data.astype(float)

    # Total strength eigenvectors
    tot_str = np.sum(s)
    reconstruct = np.zeros_like(data)

    project_matrix = text_features[indexes_max, :]
    project_matrix = np.concatenate((project_matrix, text_features[indexes_min, :]), axis = 0)
    #print(project_matrix.shape)

    # Rank K approximation of the data matrix
    #for feat in range(project_matrix.shape[0]):
    #    text_norm = project_matrix[feat] @ project_matrix[feat].T
    #    reconstruct += (data @ project_matrix[feat])[:, np.newaxis] * project_matrix[feat][np.newaxis, :] / (text_norm)**2

    #reconstruct = (data @ project_matrix.T @ project_matrix) # (dot product with text_features) dot text_features

    A = solve_non_negative_least_squares(torch.from_numpy(data).to(device), torch.from_numpy(project_matrix).to(device))
    A = A.cpu().detach().numpy()
    # Least Square (data - A @ project_matrix) = 0 <-> A = data @ project_matrix.T @ (project_matrix @ project_matrix.T)^-1
    #A = data @ project_matrix.T @ np.linalg.pinv(project_matrix @ project_matrix.T)
    
    reconstruct = A @ project_matrix
    
    # Reconstruct
    results = []
    for i, (idx_max, idx_min) in enumerate(zip(indexes_max, indexes_min)):
        results.append({"text": texts[idx_max], "strength_abs": s[i].astype(float), "strength_rel": (100 * s[i] / tot_str.astype(float))})
        results.append({"text": texts[idx_min], "strength_abs": s[i].astype(float), "strength_rel": (100 * s[i] / tot_str.astype(float))})


    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object

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
    writer = SummaryWriter("logs_test")
    print(f"\nLayer [{layer}], Head: {head}")
    # Define tag prefixes for this layer, head, and seed
    tag_prefix = f"Dataset_{dataset}/Layer_{layer}/Head_{head}/Seed_{seed}"
    
    # Center text and image data
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    
    # Project text_features to data lower-rank eigenspace with required rank
    u , s, vh = torch.linalg.svd(data, full_matrices=False)
    # Write rank of matrix
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine rank where cumulative variance exceeds 99% of total variance
    threshold = 0.99
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1

    vh = vh[:rank]
    text_features = text_features @ vh.T @ vh
    text_features = text_features / torch.linalg.norm(text_features, axis=-1)[:, np.newaxis]
    simil_matrix = (text_features @ text_features.T) # Nxd * dxM, cos similarity on each row
    # Initialize A with required gradient and with initial range guess
    scale = data.max()
    A = torch.clamp(scale*torch.rand(data.shape[0], text_features.shape[0], requires_grad=True, device=device) + \
                    2*scale, 2*scale, 3*scale)
    A_ = A.clone().detach().requires_grad_(True)

    # Set up optimizer and parameters
    optimizer = torch.optim.Adam([A_], lr=0.001)
    epochs_main = 20000
    
    # Initial ratio bewteen regularization and rmse loss 
    lbd_l1 = 1
    ratio = 1

    ## First part: main optimization loop
    patience = 500  # Number of epochs to wait for improvementy
    # Initialize variables for early stopping
    prev_cum_sum = None
    prev_indexes = torch.tensor([x for x in range(iters)], device=device)
    prev_relative_strength = None
    prev_loss = None
    stabilization_window = 50  # Number of iterations to check stability
    cum_sum_stable_count = 0  # Counter for indexes change stability
    relative_strength_stable_count = 0  # Counter for relative strength stability
    stabilization_threshold_cum = int(min(iters, rank)) + 1 # Percentage

    stabilization_threshold_strength = 0.01    
    # Training loop with early stopping
    for epoch in range(epochs_main):
        optimizer.zero_grad()  # Clear gradients from previous step

        # Clip paramteres to keep them positive
        A = torch.nn.functional.gelu(A_)
        # Compute the product A @ text_features using only stronger "iters" text with highest std across data
        text_features_std = A.std(axis=0)
        indexes = torch.argsort(text_features_std, descending=True)[:iters]
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
            continue
        
        # Backpropagation
        loss.backward()
        
        # Update A using the optimizer
        optimizer.step()

        # Compute metrics for early stopping
        text_str = text_features_std[indexes]
        tot_str = torch.sum(text_str)
        all_str = torch.sum(text_features_std)
        relative_strength = 100 * tot_str / all_str
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

            writer.add_histogram(f"{tag_prefix}/indexes", indexes, epoch)

            writer.add_histogram(f"{tag_prefix}/top_std_features_rel", 100*text_features_std[indexes][:iters]/tot_str, epoch)

        # Check stability for `cum_sum`
        if prev_cum_sum is not None:
            cum_sum_change = cum_sum 
            if cum_sum_change <= 1:
                cum_sum_stable_count += 1
            else:
                cum_sum_stable_count = max(0, cum_sum_stable_count - 1)   # Reset if not stable

        # Check stability for `relative_strength`
        if prev_relative_strength is not None:
            relative_strength_change = abs((relative_strength - prev_relative_strength) / prev_relative_strength * 100)
            if relative_strength_change < stabilization_threshold_strength:
                relative_strength_stable_count += 1
            else:
                relative_strength_stable_count = 0  # Reset if not stable

        # Update previous values
        prev_cum_sum = cum_sum
        prev_indexes = indexes
        prev_relative_strength = relative_strength

        # Check if both metrics are stable for the required window
        if cum_sum_stable_count >= stabilization_window and relative_strength_stable_count >= stabilization_window:
            print(f"Early stopping at epoch {epoch} due to stabilization.")
            break

    # Log text features
    text_str = text_features_std[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_std)   
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

    # Take columns of A with highest std (i.e. more active columns -> more active text embedding)
    text_features_std = A.std(axis=0)
    indexes = torch.argsort(text_features_std, descending=True)[:iters]
    A = A[:, indexes].detach().clone().requires_grad_(True)
    text_features = text_features[indexes, :].detach().clone().requires_grad_(True)

    ## Second part: finetune over rmse loss
    # Training loop with early stopping
    patience_counter = 0
    patience = 100  # Number of epochs to wait for improvement
    min_delta = 1e-9  # Minimum improvement in loss to be considered
    A = A.requires_grad_(True) 
    A.data.clamp_(0)

    optimizer = torch.optim.Adam([A], lr=0.001)  # Recreate the optimizer
    epochs = 500
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Clear gradients from previous step
        optimizer.zero_grad()  
        # Make prediction
        pred = A @ text_features
        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred-data)**2))
        loss = loss_rmse
        # Backpropagation
        loss.backward()
        # Update A using the optimizer
        optimizer.step()
        # Clip parameteres to keep them positive
        A.data.clamp_(0)

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
    text_str = text_features_std[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_std)   
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
    text_str = text_features_std[indexes].cpu()
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

    
@torch.no_grad()
def als_data_approx(data, text_features, texts, iters, rank, device):
    """
    This function performs als-based approximation of the attention head matrix
    using the provided text features. It starts from basis of SVD

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

    # Svd of attention head matrix (mean centered)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    text_features = text_features - mean_values_text
    data = data - mean_values_att
    # Subtract the mean from each column
    u , s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:iters]

    # Find closest unique text embedding to a given matrix U (assumed not normalized) using cosing similarity
    def project_V(V, text_features):
        # Return the closest text_features in eigen space of data matrix of top iters eigenvector
        simil_matrix = (V / np.linalg.norm(V, axis=-1)[:, np.newaxis])  @ \
                        (text_features.T / np.linalg.norm(text_features, axis=-1)[:, np.newaxis].T)# Nxd * dxM, cos similarity on each row
        indexes = np.squeeze(simil_matrix.argmax(axis=-1).astype(int))

        # Replace duplicates texts
        used_elements, used_indexes = np.unique(indexes, return_index=True)
        idxs_not_unique = np.setdiff1d(np.arange(len(indexes)), used_indexes)
        for idx_not_unique in idxs_not_unique:
            # Get argsort to find indices of max elements in descending order
            row_argsorted = simil_matrix[idx_not_unique].argsort()[::-1]

            # Find the first argmax that hasn't been used yet
            for index in row_argsorted:
                if index not in used_elements:
                    indexes[idx_not_unique] = index
                    used_elements = np.append(used_elements, index)
                    # Replace subsequent duplicates (i.e. give more priority to the first eigenvectors
                    # similarity)
                    used_elements, used_indexes = np.unique(indexes, return_index=True)
                    idxs_not_unique = np.setdiff1d(np.arange(len(indexes)), used_indexes)
                    break
        
        used_elements, used_indexes = np.unique(indexes, return_index=True)
        idxs_not_unique = np.setdiff1d(np.arange(len(indexes)), used_indexes)

        return text_features[indexes, :], indexes

    # Get the projection of text embeddings into head activations matrix space
    text_features = text_features @ vh.T @ vh
    project_matrix, indexes = project_V(vh, text_features)

    print("Starting ALS")
    ## ALS ##
    lmbda = 0.1 # Regularisation weight to make matrix denser
    n_epochs = 500 # Number of epochs
    thr = 10
    n_iters_U = 10 # Every some iterations clip U
    n_iters_V = 20 # Every some iterations project V to closest text embedding
    U = np.clip(data @ project_matrix.T @ np.linalg.pinv(project_matrix @ project_matrix.T), 0 , None) # Initial guess for U N X K add max value as highest eigenvalue
    V = project_matrix.T  # Initial guess is to use closest text to eigenvectors D X K
    # One step of als
    def als_step(target, solve_vecs, fixed_vecs, lmbda):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T @ fixed_vecs + np.diag(np.max(solve_vecs, axis = 0)) * lmbda
        b = target @ fixed_vecs
        A_inv = np.linalg.inv(A)
        solve_vecs = b @ A_inv
        return solve_vecs
    
    # Calculate the RMSE
    def rmse(data ,U,V):
        return np.sqrt(np.sum((data - U @ V.T)**2))

    # Uset different train and test errors arrays so I can plot both versions later
    train_errors_fast = []
    # Repeat until convergence
    for epoch in range(n_epochs):
        
        if epoch + thr > n_epochs: # If last epochs always keep V fixed on a projection
            # Fix V and estimate U
            U = als_step(data, U, V, lmbda=lmbda)
            U = np.clip(U, 0 , None) 
            print("Fixing V")
            # Fix U and estimate V
            V = als_step(data.T, V, U, lmbda=0.001*lmbda)  
            V_T, indexes = project_V(V.T, text_features)
            V = V_T.T
        else:
            U = als_step(data, U, V, lmbda=lmbda)
            V = als_step(data.T, V, U, lmbda=0.001*lmbda)

            if (epoch + 1) % n_iters_U == 0:
                U = np.clip(U, 0 , None) # Force U to be positive with max value as highest eigenvalue
            # Project V to closest text embedding
            if (epoch + 1) % n_iters_V == 0:
                print("Projecting V")
                V_T, indexes = project_V(V.T, text_features)
                V = V_T.T

        # Error
        train_rmse = rmse(data, U, V)
        train_errors_fast.append(train_rmse)

        print("[Epoch %d/%d] train error: %f" %(epoch+1, n_epochs, train_rmse))
        
    reconstruct = U @ V.T

    # Get total strength of text embedding basis as an average
    text_str = np.mean(U,axis=0) # Strength of a text embedding across its contributions
    tot_str = np.sum(text_str) # Total strength of text embeddings
    sort = np.argsort(text_str)[::-1]
    text_str = text_str[sort]
    indexes = indexes[sort]    
    results = [{"text": texts[idx], "strength_abs": text_str[i].astype(float), "strength_rel": (100 * text_str[i] / tot_str).astype(float)} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object
