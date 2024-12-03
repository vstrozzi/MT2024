
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

def spih_data_approx_det(data, text_features, texts, layer, head, seed, dataset, nr_basis_elem, device):
    """
    This function finds a sparse (nr_basis_elem) non-negative approximation of the attention head matrix
    using the provided text features. (i.e. A @ text_features = data s.t. A > 0  and A sparse)

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        seed: The current seed of the text dataset.
        dataset": The current text dataset used.
        nr_basis_elem: Number of iterations to perform.
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix using the found basis.
        results: Jsonline file containing the found basis and metadata.
    """
    print(f"\nLayer [{layer}], Head: {head}")

    # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=True)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top eigenvectors of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]
    nr_basis_elem = 2*rank

    # Project text_features to data lower-rank eigenspace (removes redundant informations)
    text_features = text_features @ vh.T
    # Get initialization parameters for the coefficient matrix of A
    
    # Initialize A with required gradient and with initial range guess
    data_rec = u @ torch.diag_embed(s)

    total_variance = torch.trace(data_rec.T @ data_rec)
    # Do not normalize text features
    sim_matrix_text = text_features @ text_features.T

    # Bring back on order of prediction
    # A = 2*torch.max(data, dim=-1).values.unsqueeze(-1)*A

    reconstruct = torch.zeros_like(data_rec)

    indexes = torch.empty(nr_basis_elem, device=device)
    strength = torch.empty(nr_basis_elem + 1, device=device)
    strength[0] = total_variance
    strength_diff = 0
    # Reconstruct attention head matrix by using projection on nr. iters max variance texts embeddings
    for i in range(nr_basis_elem):

        # Projects each data point (rows in data) into the feature space defined by the text embeddings.
        # Each row in projection now represents how each attention head activation vector i aligns with each text embedding j (i, j),
        # quantifying the contribution of each text_feature to the data in this iteration.

        cosine_similarity = (data_rec ) @ (text_features).T # Nxd * dxM, cos similarity on each row 

        projection_mean = cosine_similarity.mean(axis=0)
    
        # Take top text embedding with max variance for the data matrix
        top_n = torch.argmax(projection_mean)
        # Save index of text embedding
        indexes[i] = top_n

        # Remove contribution from projection only if cosine is positive
        cosine_similarity_top_n = cosine_similarity[:, top_n]
        positive_mask = cosine_similarity_top_n > 0
        # Rank 1 approximation 
        text_norm = text_features[top_n] @ text_features[top_n].T
        rank_1_approx = torch.zeros_like(data_rec)
        rank_1_approx[positive_mask] = (data_rec[positive_mask] @ text_features[top_n] / text_norm)[:, np.newaxis] \
                        * text_features[top_n][np.newaxis, :]
        reconstruct += rank_1_approx

        # Remove contribution from data matrix
        data_rec = data_rec - rank_1_approx


        strength[i+1] = torch.trace(data_rec.T @ data_rec)
        print(strength[i+1].item())
    
    results = [{"text": texts[int(idx.item())], "strength_abs": (strength[i] - strength[i+1]).item(), "strength_rel": (100 * (strength[i] - strength[i+1]) / total_variance).item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }

    for result in results:
        print(result)


    return reconstruct @ vh + mean_values_att, json_object

def highest_cos_sim_head(data, text_features, texts, layer, head, seed, dataset, nr_basis_elem, device):
    """
    This function finds a sparse (nr_basis_elem) non-negative approximation of the attention head matrix
    using the provided text features. (i.e. A @ text_features = data s.t. A > 0  and A sparse)

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        seed: The current seed of the text dataset.
        dataset": The current text dataset used.
        nr_basis_elem: Number of iterations to perform.
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix using the found basis.
        results: Jsonline file containing the found basis and metadata.
    """
    print(f"\nLayer [{layer}], Head: {head}")

    # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=True)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top eigenvectors of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]
    nr_basis_elem = 2*rank

    # Project text_features to data lower-rank eigenspace (removes redundant informations)
    text_features = text_features @ vh.T
    # Get initialization parameters for the coefficient matrix of A
    
    # Initialize A with required gradient and with initial range guess
    # Cosine similarity
    data_rec = u @ torch.diag_embed(s)
    data_rec = data_rec

    # Normalize text features
    # text_features = text_features / torch.linalg.norm(text_features, axis=-1)[:, np.newaxis]

    reconstruct = torch.zeros_like(data_rec)

   
    cosine_similarity = (data_rec ) @ (text_features ).T # Nxd * dxM, cos similarity on each row 

    projection_mean = cosine_similarity.mean(axis=0)

    indexes = torch.argsort(projection_mean, descending=True)[:nr_basis_elem]

    results = [{"text": texts[int(idx.item())], "strength_abs": projection_mean[i].item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }

    for result in results:
        print(result)


    return reconstruct @ vh + mean_values_att, json_object


def svd_data_approx(data, text_features, texts, layer, head, seed, iters, rank, device):
    print(f"\nLayer [{layer}], Head: {head}")

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
     # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)
    # Subtract the mean from each column
    u, s, vh = np.linalg.svd(data, full_matrices=False)

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=True)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top eigenvectors of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]

    nr_basis_elem = rank
    nr = 5
    # Get the projection of text embeddings into head activations matrix space
    text_features = text_features @ vh.T @ vh
    
    # Return the closest text_features in eigen space of data matrix of top iters eigenvector
    simil_matrix = (vh @ text_features.T) # Nxd * dxM, cos similarity on each row
    indexes_max = torch.squeeze(torch.argsort(simil_matrix, dim=-1))[:nr_basis_elem, :nr]
    indexes_min = torch.squeeze(torch.argsort(simil_matrix, dim=-1, descending=True))[:nr_basis_elem, :nr]

    # Replace duplicates texts
    """ used_elements_max, used_indexes_max = np.unique(indexes_max, return_index=True)
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
                break """


    # Total strength eigenvectors
    tot_str = torch.sum(s)
    reconstruct = torch.zeros_like(data)

    project_matrix = text_features[indexes_min[:, 0], :]

    # Least Square (data - A @ project_matrix) = 0 <-> A = data @ project_matrix.T @ (project_matrix @ project_matrix.T)^-1
    A = data @ project_matrix.T @ np.linalg.pinv(project_matrix @ project_matrix.T)
    
    reconstruct = A @ project_matrix
    
    # Reconstruct
    results = []
    for i, (idx_max, idx_min) in enumerate(zip(indexes_max, indexes_min)):
        text = []
        for k in range(nr):
            text.append(texts[idx_max[k].item()])
        for k in range(nr):
            text.append(texts[idx_min[k].item()])

        results.append({"text": text, "strength_abs": s[i].item(), "strength_rel": (100 * s[i] / tot_str).item()})   # Reconstruct original matrix with new basis


    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }

    for result in results:
        print(result)

    return reconstruct + mean_values_att, json_object
