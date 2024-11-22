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
from utils.misc import accuracy

from utils.algorithms_text_set import * # All the algorithms


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
        "--text_size",
        type=int,
        default=3500,
        help="The number of texts to consider in total",
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
    # Take subset of lines
    lines = lines[-args.text_size:]
    text_features = text_features[-args.text_size:]
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_completeness_{args.text_descriptions}_top_{args.texts_per_head}_heads_{args.model}.jsonl",
        ),
        "w",
    ) as jsonl_file:
        # Used algorithm
        select_algo = splice_data_approx
        # Compute text span per head and approximate its output by projecting each activation to the span of its text.
        # Evaluate the accuracy of the model on the given dataset.
        for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]): # for the selected layers
            for head in range(attns.shape[2]): # for each head in the layer
                results, texts, json_info = text_span(
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

                # Json of our data
                json_object = {
                    "layer": i,
                    "head": head,
                    **json_info
                }

                jsonl_file.write(json.dumps(json_object) + "\n")

        # Get total contribution of the model
        mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))
        # Compute another iteration of the selected algorithm for the final output
        _, texts, json_info = select_algo(
                    mean_ablated_and_replaced,
                    text_features,
                    lines,
                    args.texts_per_head,
                    args.w_ov_rank,
                    args.device,
                )        
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
        # Json of the final embedding
        json_object = {
            "layer": -1,
            "head": -1,
            "accuracy": current_accuracy,
            "nr_texts": len(all_texts),
            **json_info
        }
        jsonl_file.write(json.dumps(json_object))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
