import numpy as np
import os.path
import argparse
from pathlib import Path

from utils.misc.misc import accuracy


def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

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
        "--figures_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="imagenet, waterbirds, cub, binary_waterbirds",
    )
    return parser


def main(args):
    """
    Need to run compute_prs.py for the given model before running this script.
    Compute the final embedding of the given model.
    """
    attns = np.load(os.path.join(args.output_dir, f"{args.dataset}_attn_{args.model}_seed_{args.seed}.npy"), mmap_mode="r")  # [b, l, h, d]
    mlps = np.load(os.path.join(args.output_dir, f"{args.dataset}_mlp_{args.model}_seed_{args.seed}.npy"), mmap_mode="r")  # [b, l+1, d]
    
    print(attns.shape, mlps.shape)
    # Final embeddings of the model, from summing all the activations
    # of the model(i.e. reconstruct normal output of ViT).
    final_embeddings = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    
    with open(
    os.path.join(args.output_dir, f"{args.dataset}_embeddings_{args.model}_seed_{args.seed}.npy"), "wb"
    ) as f:
        np.save(f, final_embeddings)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)
