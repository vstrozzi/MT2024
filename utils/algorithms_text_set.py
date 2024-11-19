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
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path
from torch.nn import functional as F

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
    text_features = text_features - np.mean(text_features, axis=0)
    mean_values_att = np.mean(data, axis=0)
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

    return reconstruct + mean_values_att, results

@torch.no_grad()
def svd_data_approx(data, text_features, texts, iters, rank, device):
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
    text_features = text_features - np.mean(text_features, axis=0)
    data = data - mean_values_att
    # Subtract the mean from each column
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:iters]

    # Get the projection of text embeddings into head activations matrix space
    text_features = text_features @ vh.T @ vh

    # Return the closest text_features in eigen space of data matrix of top iters eigenvector
    simil_matrix = (vh @ text_features.T)/ np.linalg.norm(text_features, axis=-1)[:, np.newaxis].T # Nxd * dxM, cos similarity on each row
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


    # Fix unique texts per eigenvector
    data = data.astype(float)

    # Total strength eigenvectors
    tot_str = np.sum(s)
    results = [texts[idx] + ", with " + str(s[i]) + " on " + str(tot_str) + " (" + str(100 * s[i] / tot_str) + ")" for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis
    reconstruct = np.zeros_like(data)

    project_matrix = text_features[indexes, :]

    # Rank K approximation of the data matrix
    #for feat in range(project_matrix.shape[0]):
    #    text_norm = project_matrix[feat] @ project_matrix[feat].T
    #    reconstruct += (data @ project_matrix[feat])[:, np.newaxis] * project_matrix[feat][np.newaxis, :] / (text_norm)**2
    # reconstruct = ((data @ project_matrix.T) / np.square((np.linalg.norm(project_matrix, axis=-1)[:, np.newaxis].T)) \
    #                 @ project_matrix) # (dot product with text_features) dot text_features

    # Least Square (data - A @ project_matrix) = 0 <-> A = data @ project_matrix.T @ (project_matrix @ project_matrix.T)^-1
    A = data @ project_matrix.T @ np.linalg.pinv(project_matrix @ project_matrix.T)
    
    reconstruct = A @ project_matrix + mean_values_att
    return reconstruct, results 

def splice_data_approx(data, text_features, texts, iters, rank, device):
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

    # Fix unique texts per eigenvector
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Project text_features to data eigenspaces with required rank
    u , s, vh = torch.linalg.svd(data, full_matrices=False)
    vh = vh[:rank]
    text_features = text_features @ vh.T @ vh
    simil_matrix = (text_features @ text_features.T) # Nxd * dxM, cos similarity on each row

    # Initialize A with required gradient
    A = torch.clamp(torch.rand(data.shape[0], text_features.shape[0], requires_grad=True) , 0, None).to(device)
    A = A.clone().detach().requires_grad_(True)

    # Set up optimizer and parameters
    optimizer = torch.optim.Adam([A], lr=0.01)
    epochs = 1500
    lbd_l1 = 0.005

    patience = 1500  # Number of epochs to wait for improvement
    best_loss = float('inf')
    patience_counter = 0
    # Range of splitting values to consider
    interv = 2

    steps = [A.shape[0], A.shape[0], A.shape[0], iters] #np.linspace(A.shape[0], iters, interv).astype(int).copy()

    print(steps)
    c = 0
    # Training loop with early stopping
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients from previous step

        # Consider a subset of the possible text features
        selected = steps[c]
        # Compute the product A @ text_features using only stronger iters entries with highest std across entries
        text_features_std = A.std(axis=0)
        #indexes = torch.argsort(text_features_std, descending=True)
        indexes_selected = torch.argsort(text_features_std, descending=True)[:iters]

        # Favour texes which have highest similarity to each others
        loss_cosine = -simil_matrix[indexes_selected, indexes_selected].mean()
        pred = A[:, indexes_selected] @ text_features[indexes_selected, :]
        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred-data)**2))

        # Regularization L1 on row (i.e. sparse row i.e. few text embeddings)
        # Only for top used, yielding to anice
        loss_l1 = lbd_l1 * torch.norm(A[:, indexes_selected], p=1, dim=1).mean()

        loss = loss_l1 + loss_rmse + loss_cosine
        # Backpropagation
        loss.backward()
        
        # Update A using the optimizer
        optimizer.step()

        # Clip paramteres to keep them positive
        A.data.clamp_(0)

        # Print loss every 100 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        # Take the next subset of textes to keep optimizing on
        if c != len(steps) - 1 and epoch > (c + 1) * epochs//interv:
            c += 1 


        # Early stopping logic
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0  # Reset patience counter when improvement occurs
        else:
            patience_counter += 1  # Increment if no improvement

        # Stop training if patience is exhausted
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} with best loss: {best_loss:.6f}")
            break
    
    print(data.max())
    print(pred.max())
    print(torch.sqrt(torch.nn.functional.mse_loss(pred, data)))
    # Take columns of A with highest std (i.e. more active columns -> more active text embedding)
    text_features_std = A.std(axis=0).detach().cpu().numpy() 
    indexes = np.argsort(text_features_std)[::-1][:iters].copy()
    # Retrieve corresponding text
    text_str = text_features_std[indexes]
    tot_str = np.sum(text_str) # Total strength of text embeddings on that
    print("We have a value of %f for selected strength" % tot_str)
    print("We have a total strength of %f for all the columns" % np.sum(text_features_std))
    results = [{"text": texts[idx], "strength_abs": text_str[i].astype(float), "strength_rel": (100 * text_str[i] / tot_str).astype(float)} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Compute reconstruction using only our selected 
    A = A.clamp(0, None)
    reconstruct = A[:, indexes].detach().cpu().numpy() @ text_features[indexes, :].detach().cpu().numpy()
    
    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, np.array(texts)[indexes], json_object

    
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
    text_features = text_features - np.mean(text_features, axis=0)
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
    lmbda = 0.5 # Regularisation weight to make matris
    n_epochs = 100 # Number of epochs
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
        
    reconstruct = U @ V.T + mean_values_att

    # Get total strength of text embedding basis as an average
    text_str = np.mean(U,axis=0) # Strength of a text embedding across its contributions
    tot_str = np.sum(text_str) # Total strength of text embeddings
    sort = np.argsort(text_str)[::-1]
    text_str = text_str[sort]
    indexes = indexes[sort]
    results = [texts[idx] + ", with " + str(text_str[i]) + " on " + str(tot_str) + " (" + str(100 * text_str[i] / tot_str) + ")" for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis
    return reconstruct, results 
