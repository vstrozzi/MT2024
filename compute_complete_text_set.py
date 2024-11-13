import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os
import einops
import sympy
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path
from torch.nn import functional as F

from utils.misc import accuracy


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

    # Initialize A with required gradient
    A = torch.clamp(torch.randn(data.shape[0], text_features.shape[0], requires_grad=True), 0, None).to(device)
    A = A.clone().detach().requires_grad_(True)

    # Set up optimizer and parameters
    optimizer = torch.optim.Adam([A], lr=0.01)
    epochs = 400
    lbd = 0.001
    patience = 20  # Number of epochs to wait for improvement
    best_loss = float('inf')
    patience_counter = 0

    # Training loop with early stopping
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients from previous step

        # Compute the product A @ text_features
        A_clamp = torch.clamp(A, 0, None)
        pred = torch.matmul(A_clamp, text_features)
        
        # Compute the mean squared error loss
        loss = torch.nn.functional.mse_loss(pred, data)
        
        # Regularization L1 (optional, uncomment if needed)
        loss += lbd/(epoch + 1) * torch.abs(A_clamp.sum(dim=-1)).sum()
        
        # Backpropagation
        loss.backward()
        
        # Update A using the optimizer
        optimizer.step()
        
        # Print loss every 100 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

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


    reconstruct = A.detach().cpu().numpy() @ text_features.detach().cpu().numpy()

    # Take columns of A with highest mean
    text_features_mean = A.mean(axis=0).detach().cpu().numpy() 
    # Take top text embedding with max variance for the data matrix
    indexes = np.argsort(text_features_mean)[::-1][:iters]
    text_str = text_features_mean[indexes]
    tot_str = np.sum(text_str) # Total strength of text embeddings
    results = [texts[idx] + ", with " + str(text_str[i]) + " on " + str(tot_str) + " (" + str(100 * text_str[i] / tot_str) + ")" for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    return reconstruct + mean_values_att, results

    
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
    _, s, vh = np.linalg.svd(data, full_matrices=False)
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
    lmbda = 0.0001 # Regularisation weight to make matris
    n_epochs = 100 # Number of epochs
    thr = 50
    n_iters_U = 10 # Every some iterations clip U
    n_iters_V = 20 # Every some iterations project V to closest text embedding
    U = np.clip(data @ project_matrix.T, 0 , None) # Initial guess for U N X K add max value as highest eigenvalue
    V = project_matrix.T  # Initial guess is to use closest text to eigenvectors D X K
    # One step of als
    def als_step(target, solve_vecs, fixed_vecs, lmbda):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T @ fixed_vecs + np.eye(iters) * lmbda
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
            if (n_epochs + 1) % n_iters_U == 0:
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

def get_args_parser():
    parser = argparse.ArgumentParser("Completeness part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,   
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--text_descriptions",
        default="image_descriptions_per_class",
        type=str,
        help="name of the evalauted text set",
    )
    parser.add_argument(
        "--text_dir",
        default="./text_descriptions",
        type=str,
        help="The folder with the text files",
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet or waterbirds"
    )
    parser.add_argument(
        "--num_of_last_layers",
        type=int,
        default=4,
        help="How many attention layers to replace.",
    )
    parser.add_argument(
        "--w_ov_rank", type=int, default=80, help="The rank of the OV matrix"
    )
    parser.add_argument(
        "--texts_per_head",
        type=int,
        default=20,
        help="The number of text examples per head.",
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    """
    Evaluate a CLIP representation for a given dataset of text. This is needed to run text_span algorithm.
    """
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), "rb"
    ) as f:
        mlps = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)
    
    labels = np.load(os.path.join(args.input_dir, f"{args.dataset}_labels_{args.model}.npy")) 

    print(f"Number of layers: {attns.shape[1]}")
    all_texts = set()
    
    # Mean-ablate the other parts 
    for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers):
        for head in range(attns.shape[2]):
            attns[:, i, head] = np.mean(attns[:, i, head], axis=0, keepdims=True)
    # Load text descriptions:
    with open(
        os.path.join(args.input_dir, f"{args.text_descriptions}_{args.model}.npy"), "rb"
    ) as f:
        text_features = np.load(f)
    with open(os.path.join(args.text_dir, f"{args.text_descriptions}.txt"), "r") as f:
        lines = [i.replace("\n", "") for i in f.readlines()]
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_completeness_{args.text_descriptions}_top_{args.texts_per_head}_heads_{args.model}.txt",
        ),
        "w",
    ) as w:
        # Compute text span per head and approximate its output by projecting each activation to the span of its text.
        # Evaluate the accuracy of the model on the given dataset.
        for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]): # for the selected layers
            for head in range(attns.shape[2]): # for each head in the layer
                results, texts = splice_data_approx(
                    attns[:, i, head],
                    text_features,
                    lines,
                    args.texts_per_head,
                    args.w_ov_rank,
                    args.device,
                )
                # Use the final reconstructed attention head matrix
                attns[:, i, head] = results
                all_texts |= set(texts)
                w.write(f"------------------\n")
                w.write(f"Layer {i}, Head {head}\n")
                w.write(f"------------------\n")
                for text in texts:
                    w.write(f"{text}\n")

        # Get total contribution of the model
        mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))
        # Get final clip output
        projections = torch.from_numpy(mean_ablated_and_replaced).float().to(
            args.device
        ) @ torch.from_numpy(classifier).float().to(args.device)
        current_accuracy = (
            accuracy(projections.cpu(), torch.from_numpy(labels))[0] * 100.0
        )
        print(
            f"Current accuracy:",
            current_accuracy,
            "\nNumber of texts:",
            len(all_texts),
        )
        w.write(f"------------------\n")
        w.write(
            f"Current accuracy: {current_accuracy}\nNumber of texts: {len(all_texts)}"
        )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
