import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import tqdm
import argparse
from utils.factory import create_model_and_transforms, get_tokenizer
import re
import heapq
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from torch.nn import functional as F
import einops
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

def top_k_contributions_naive(head_texts, k, attns, topic_embedding, model, tokenizer, device):
    # Initialize a max heap for top-k similarities
    k = args.nr_components  # specify the number of top elements you want
    top_k_heap = []

    # Evaluate cosine similarity of each head components
    for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]): # for the selected layers
        for h in range(attns.shape[2]): # for each head
            with torch.no_grad():
                texts = head_texts[i, h]
                texts = tokenizer(texts).to(device)
                for j, text in enumerate(texts):
                    text_embeddings = model.encode_text(text.unsqueeze(0))

                    text_embeddings = F.normalize(text_embeddings, dim=-1)
                    text_embeddings /= text_embeddings.norm()

                    # Calculate cosine similarity
                    similarity = (text_embeddings @ topic_embedding.T).item()

                    # Push similarity value along with indices to heap, maintaining the top-k elements
                    heapq.heappush(top_k_heap, (similarity, (i, h, j)))

                    if len(top_k_heap) > k:
                        heapq.heappop(top_k_heap)

    # Retrieve the top k values from the heap, sorted by similarity
    return sorted(top_k_heap, key=lambda x: -x[0])

# Negative cosine function
def objective_function(vars, mat_embedd, topic_embedding):
    print(cosine_similarity((np.expand_dims(vars, 0) @ mat_embedd), topic_embedding))
    return -cosine_similarity((np.expand_dims(vars, 0) @ mat_embedd), topic_embedding)
    
# Get top k contributions using sequential least square quadratic programming
def top_k_contributions(head_texts, k, attns, topic_embedding, model, tokenizer, device):
    # Derive a row matrix of all the possible text embeddings
    l_start = attns.shape[1] - args.num_of_last_layers
    l_end = attns.shape[1]
    # Get text Embedding
    texts = head_texts[l_start:l_end]
    texts = einops.rearrange(texts, 'l h t -> (l h t)')

    texts = tokenizer(texts).to(device)
    texts_embeddings = None
    # Get encoding in batches
    with torch.no_grad():
        for batch_texts in iter(texts.split(10)):
            batch_texts_embeddings = model.encode_text(batch_texts)
            if texts_embeddings is None:   
                texts_embeddings = batch_texts_embeddings
            else:
                texts_embeddings = torch.cat((texts_embeddings, batch_texts_embeddings), dim=0)
    
    # Normalize embeddings
    texts_embeddings = F.normalize(texts_embeddings, dim=-1).cpu().detach().numpy()
    topic_embedding = topic_embedding.cpu().detach().numpy()
    # Non linear programming
    constraint = {"type": "eq", "fun": lambda x: np.sum(x) - k}
    bounds = [(0, 1) for _ in range(texts_embeddings.shape[0])]
    init_guess = np.array([1 if i < k else 0 for i in range(texts_embeddings.shape[0])])
    result = minimize(objective_function, init_guess, args=(texts_embeddings, topic_embedding), bounds=bounds, constraints=constraint, options={'maxiter':20})
    # Get the indexes of the top k elements
    indexes = np.argsort(result.x)[-k:][::-1]
    values = einops.rearrange(result.x, '(l h t) -> l h t', l=args.num_of_last_layers, h=attns.shape[2]) 
    values_max_idxs = [np.unravel_index(idx, values.shape) for idx in indexes]
    top_k = [(result.x[idx], (l + l_start, h, text_pos)) for idx, (l, h, text_pos) in zip(indexes, values_max_idxs)] 

    return top_k


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
        default=4,
        help="Layers you want to look at.",
    )

    parser.add_argument(
        "--topic",
        type=str,
        default="The zoom-up main object in the image.",
        help="The topic we want to look at.",
    )
    parser.add_argument(
        "--head_descriptions",
        type=str,
        default="imagenet_completeness_image_descriptions_general_top_30_heads_ViT-B-32.txt",
        help="The file containing the text explanations per head.",
    )

    parser.add_argument("--nr_components", type=int, default=5, help="Number of components to use for the topic")

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
    topic_embedding = F.normalize(topic_embedding, dim=-1)
    
    # Get the topic per head
    head_nr = int(re.search(r'_top_(\d+)_heads_', args.head_descriptions).group(1))
    head_texts = read_head_descriptions(head_descriptions,  attns.shape[1], attns.shape[2], head_nr)
    
    # Get the top k contributions
    top_k_results = top_k_contributions(head_texts, args.nr_components, attns, topic_embedding, model, tokenizer, args.device)

    # Write the top k similarities in order, with corresponding texts
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_with_nr_elem_head_{head_nr}_and_{args.nr_components}_components_for_topic_{args.topic}_model_{args.model}.txt",
        ),
        "w",
    ) as w:
        for similarity, (layer, head, text) in top_k_results:
            w.write(f"Layer: {layer}, Head: {head}, Rank: {text}, Similarity: {similarity}, Text: {head_texts[layer, head, text]}\n")
    
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
