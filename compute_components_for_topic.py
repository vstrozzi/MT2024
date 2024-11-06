import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import tqdm
import argparse
from utils.factory import create_model_and_transforms, get_tokenizer
import re
import heapq

from pathlib import Path
from torch.nn import functional as F

from utils.misc import accuracy

def read_head_descriptions(head_descriptions, nr_layers, nr_heads_per_layer, texts_per_head):
    text_array = np.empty((nr_layers, nr_heads_per_layer, texts_per_head), dtype=object)
    # Set counters
    layer = 0
    head = -1
    line_count = 0
    layer_count = 0

    # Open and read the file
    for line in head_descriptions.readlines():
        line = line.strip()
        # Skip empty lines
        if line.startswith("------------------"):
            continue
        # Stop processing if a line with "Current accuracy" or "Number of texts" is encountered
        if line.startswith("Current accuracy") or line.startswith("Number of texts"):
            break
        # Detect layer and head
        layer_match = re.match(r'Layer (\d+), Head (\d+)', line)
        if layer_match:
            layer = int(layer_match.group(1))
            head = int(layer_match.group(2))
            line_count = 0  # Reset line counter for each new head
            layer_count += 1
            continue
        
        # If it's a data line with text
        if head >= 0 and layer >= 0:
            text_match = re.match(r'(.+?),\s*with\s*[\d.]+', line)
            if text_match:
                description = text_match.group(1).strip()
                text_array[layer, head, line_count] = description
                line_count += 1
    
    return text_array

def get_args_parser():
    parser = argparse.ArgumentParser("Completeness part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,   
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", type=str)
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )

    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet, cub or waterbirds"
    )

    parser.add_argument(
        "--num_of_last_layers",
        type=int,
        default=3,
        help="Layers you want to look at.",
    )

    parser.add_argument(
        "--topic",
        type=str,
        default="a dog.",
        help="The number of text examples per head.",
    )
    parser.add_argument(
        "--head_descriptions",
        type=str,
        default="imagenet_completeness_image_descriptions_general_top_30_heads_ViT-B-32.txt",
        help="Layers you want to look at.",
    )

    parser.add_argument("--nr_components", type=int, default=10, help="Number of components to use for the topic")

    parser.add_argument("--device", default="cpu", help="device to use for testing")
    return parser


def main(args):
    """
    Find the top basis elements across different components with higher cosine similarity
    with a given text description. We exploits the distributivity of a Dot Product.
    A dataset is needed to get the activations for each component you want to look at. 
    Run the code after you have run compute_components_for_topic.py.
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
    
    head_descriptions = open(os.path.join(args.input_dir, args.head_descriptions), "r")


    print(f"Number of layers: {attns.shape[1]}")
    # Get the model
    model, _, _ = create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = get_tokenizer(args.model)
    model.to(args.device)
    model.eval()

    # Evaluate clip embedding for the given topic text
    texts = tokenizer(args.topic).to(args.device)  # tokenize
    topic_embedding = model.encode_text(texts)
    topic_embedding = F.normalize(topic_embedding, dim=-1).mean(dim=0)
    topic_embedding /= topic_embedding.norm()
    
    # Get the topic per head
    head_nr = int(re.search(r'_top_(\d+)_heads_', args.head_descriptions).group(1))
    head_texts = read_head_descriptions(head_descriptions,  attns.shape[1], attns.shape[2], head_nr)
    

    # Initialize a max heap for top-k similarities
    k = args.nr_components  # specify the number of top elements you want
    top_k_heap = []

    # Evaluate cosine similarity of each head components
    for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]): # for the selected layers
        for h in range(attns.shape[2]): # for each head
            with torch.no_grad():
                texts = head_texts[i, h]
                texts = tokenizer(texts).to(args.device)
                for j, text in enumerate(texts):
                    text_embeddings = model.encode_text(text.unsqueeze(0))

                    text_embeddings = F.normalize(text_embeddings, dim=-1).mean(dim=0)
                    text_embeddings /= text_embeddings.norm()

                    # Calculate cosine similarity
                    similarity = (text_embeddings @ topic_embedding.T).item()

                    # Push similarity value along with indices to heap, maintaining the top-k elements
                    heapq.heappush(top_k_heap, (similarity, (i, h, j, text_embeddings)))

                    if len(top_k_heap) > k:
                        heapq.heappop(top_k_heap)

    # Retrieve the top k values from the heap, sorted by similarity
    top_k_results = sorted(top_k_heap, key=lambda x: -x[0])

    # Print the top k similarities and corresponding texts
    for similarity, (layer, head, text, text_embeddings) in top_k_results:
        print(f"Layer: {layer}, Head: {head}, Similarity: {similarity}, Text: {head_texts[layer, head, text]}")    
   
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
