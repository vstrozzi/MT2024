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

def svd_parameters_init(vh, s, text_features, rank):
    """
    This function performs SVD-based initialization of the attention head matrix
    using the provided text features. It returns the indexes of 2 text features per eigenvector, 
    which have the highest and lower cosine similarity with the top rank eigenvectors.
    (i.e. the features which spans in the most similar and opposite direction of the eigenvectors, which
    can be then used as a non-negative basis)

    Args:
        vh: The top eigenvectors of the data matrix (D = vh s u)
        s: The strength of the eigenvectors (D = vh s u)
        text_features: The text features matrix (clip embedding).
        rank: Rank of the data matrix

    Returns:
        indexes: Indexes to start with.
        weights: List of weights to start with.
    """


    # Return the closest text_features in eigen space of data matrix of top iters eigenvector
    simil_matrix = vh @ text_features.T  # Nxd * dxM, cosine similarity on each row
    indexes_max = torch.squeeze(simil_matrix.argmax(dim=-1))[:rank]
    indexes_min = torch.squeeze(simil_matrix.argmin(dim=-1))[:rank]

    # Get indexes of duplicate texts
    """ used_elements_max, used_indexes_max = torch.unique(indexes_max, return_inverse=True)

    idxs_not_unique_max = torch.diff((torch.arange(len(indexes_max)), used_indexes_max), dim=1)
    used_elements_min, used_indexes_min = torch.unique(indexes_min, return_inverse=True)
    idxs_not_unique_min = torch.diff((torch.arange(len(indexes_min)), used_indexes_min), dim=1)
    
    for idx_not_unique in idxs_not_unique_max:
        # Get argsort to find indices of max elements in descending order
        row_argsorted_desc = torch.argsort(simil_matrix[idx_not_unique], descending=True)

        # Find the first argmax that hasn't been used yet
        for index in row_argsorted_desc:
            if index not in used_elements_max:
                indexes_max[idx_not_unique] = index
                used_elements_max = torch.cat((used_elements_max, index.unsqueeze(0)))
                break

    for idx_not_unique in idxs_not_unique_min:
        # Get argsort to find indices of max elements in ascending order
        row_argsorted_asc = torch.argsort(simil_matrix[idx_not_unique])

        # Find the first argmin that hasn't been used yet
        for index in row_argsorted_asc:
            if index not in used_elements_min:
                indexes_min[idx_not_unique] = index
                used_elements_min = torch.cat((used_elements_min, index.unsqueeze(0)))
                break
    """
    
    # Total strength eigenvectors
    indexes = torch.cat((indexes_max, indexes_min))
    strength = torch.cat((s, s), dim=0)
 

    return indexes, strength