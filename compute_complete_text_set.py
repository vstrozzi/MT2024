import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os
import einops
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path

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
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:rank]
    # return np.matrix(u[:, :80]) * np.diag(s[:80]) * np.matrix(vh[:80, :]) ,results # TASK: this line to test svd
    # Get the projection of text embeddings into head activations matrix space
    text_features = (vh.T @ vh @ text_features.T).T

        
    # Mean center the data matrix
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

    return reconstruct, results

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

    # Svd of attention head matrix
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:iters]

    # Get the projection of text embeddings into head activations matrix space
    text_features = (vh.T @ vh @ text_features.T).T

    # Return the closest text_features in eigen space of data matrix of top iters eigenvector
    simil_matrix = vh @ text_features.T # Nxd * dxM, cos similarity on each row

    # Retrieve texts depending on the index
    indexes = np.squeeze(simil_matrix.argmax(axis=-1).astype(int)).tolist()
    indexes = np.array([indexes]) if isinstance(indexes, int) else indexes
    data = data.astype(float)
    # Total strength eigenvectors
    tot_str = np.sum(s)
    results = [texts[idx] + ", with " + str(s[i]) + " on " + str(tot_str) + " (" + str(100 * s[i] / tot_str) + ")" for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis
    reconstruct = np.zeros_like(data)
    # Project data matrix back and forth
    project_matrix = text_features[indexes, :]
    pseudo_inverse = np.linalg.pinv(project_matrix)
    reconstruct = (data @ project_matrix.T) @ pseudo_inverse.T

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
                results, texts = svd_data_approx(
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
