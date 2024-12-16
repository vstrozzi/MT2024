import json
from dataclasses import dataclass
from typing import List
import pandas as pd
from tabulate import tabulate
import torch
@dataclass
### Layout of data
class PrincipalComponentRecord:
    layer: int
    head: int
    princ_comp: int
    strength_abs: float
    strength_rel: float
    cosine_similarity: float
    texts: List[str]
    project_matrix: torch.Tensor
    vh: torch.Tensor
    rank: int

def get_data(attention_dataset, min_princ_comp=-1, skip_final=False):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - attention_dataset (str): The path to the JSON file containing attention data.
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).

    Returns:
    - A list of dictionaries containing details for each principal component of interest.
    """
    with open(attention_dataset, "r") as json_file:
        data = []
        for line in json_file:
            entry = json.loads(line)  # Parse each line as a JSON object, producing a dictionary-like structure
             # Skip the final clip embeddings if requested
            if skip_final and entry["head"] == -1:
                continue

            # Each entry includes a sorted list of principal components. 
            # We want to record information up to a certain minimum principal component index (min_princ_comp).
            for i, princ_comp_data in enumerate(entry["embeddings_sort"]):
                # Stop if we've reached the minimum principal component index to limit how many we gather from each entry
                if i == min_princ_comp:
                    break
               
                # Append a dictionary of details for each principal component of interest
                data.append({
                    "layer": entry["layer"],
                    "head": entry["head"],
                    "princ_comp": i,
                    "strength_abs": princ_comp_data["strength_abs"],
                    "strength_rel": princ_comp_data["strength_rel"],
                    "cosine_similarity": princ_comp_data["cosine_similarity"],
                    "texts": princ_comp_data["text"],
                    "rank": len(entry["embeddings_sort"]),
                    "project_matrix": entry["project_matrix"],
                    "vh": entry["vh"],
                    "mean_values_text": entry["mean_values_text"],
                    "mean_values_att": entry["mean_values_att"],
                })
        
    return data

def print_data(data, min_princ_comp=-1):
    """
    Print the collected data in a formatted table.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).
    Returns:
    - None
    """
    # Convert the collected data into a Pandas DataFrame for easier manipulation and printing
    top_k_df = pd.DataFrame(data)

    # Iterate over each row in the DataFrame to display details about the principal components and their associated texts
    for row in top_k_df.itertuples():
        output_rows = []
        texts = row.texts
        half_length = len(texts) // 2
        
        # Skip the enetry if above the minimum number we want
        if row.princ_comp >= min_princ_comp and min_princ_comp != -1:
            continue

        # Determine if the first half of the texts represent a positive component by checking the associated value
        is_positive_first = list(texts[0].values())[1] > 0
        
        # Split the texts into two halves: first half (positive or negative) and second half (the opposite polarity)
        positive_texts = texts[:half_length]
        negative_texts = texts[half_length:]
        
        # Pair up corresponding positive and negative texts and collect them in output_rows for tabular display
        for pos, neg in zip(positive_texts, negative_texts):
            pos_text = list(pos.values())[0]
            pos_val  = list(pos.values())[1]
            neg_text = list(neg.values())[0]
            neg_val  = list(neg.values())[1]    
            
            output_rows.append([pos_text, pos_val, neg_text, neg_val])

        # Print summary information about the current principal component:
        # Including layer, head, principal component index, absolute variance, relative variance, and head rank.
        print(f"Layer {row.layer}, Head {row.head}, Principal Component {row.princ_comp}, "
            f"Variance {row.strength_abs:.3f}, Relative Variance {row.strength_rel:.3f}, Head Rank {row.rank}")
        
        # Set the column headers based on whether the first half was considered positive
        if is_positive_first:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]
        
        # Create a DataFrame from the collected rows of positive/negative texts and print it in a formatted table
        output_df = pd.DataFrame(output_rows, columns=columns)
        print(tabulate(output_df, headers='keys', tablefmt='psql'))


def sort_data_by(data, key="strength_abs", descending=True):
    """
    Sorts data based on the 'strength_abs' key.

    Parameters:
    - data (list): List of dictionaries to sort.
    - key (str): Key to sort by.
    - descending (bool): Sort in descending order if True, ascending otherwise.

    Returns:
    - list: Sorted list of data.
    """
    return sorted(data, key=lambda x: x.get(key, 0), reverse=descending)

def top_data(data, top_k=5):
    """
    Get the top-k elements of data (already sorted)

    Parameters:
    - data (list): List of dictionaries from which to retrieve top elements.
    - top_k (int): Number of top-k elements to retrieve.

    Returns:
    - list: Top-k data.
    """
    return data[:top_k]

def map_data(data, lbd_func=None):
    """
    Apply lambda function on each element of data.

    Parameters:
    - data (list): List of dictionaries from which to retrieve top elements.
    - lbd_func (x: x): Lambda function.

    Returns:
    - list: Mapped elements.
    """
    return [lbd_func(x) for x in data]
def reconstruct_embeddings(data, embeddings, types, return_princ_comp=False):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings

    Returns:
    - list: Reconstructed embeddings.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0 or len(types) != len(embeddings):
        assert False, "No embeddings to reconstruct or different lengths."
    # Initialize the reconstructed embeddings
    recontruct_embeddings = [torch.zeros_like(embeddings[0]) for _ in range(len(embeddings))]
    
    # Iterate over each principal component of interest
    for component in data:
        # Retrieve projection matrices and mean values
        project_matrix = torch.tensor(component["project_matrix"])
        vh = torch.tensor(component["vh"])
        mean_values_text = torch.tensor(component["mean_values_text"])
        mean_values_images = torch.tensor(component["mean_values_att"])
        princ_comp = component["princ_comp"]

        mask = torch.zeros((vh.shape[0]))
        # Reconstruct Embeddings
        for i in range(len(embeddings)):
            mask = torch.zeros(vh.shape[0])
            if types[i] == "text":
                emb_cent = embeddings[i] - mean_values_text
                mask[princ_comp] = (emb_cent @ vh.T)[:, princ_comp].squeeze()
                recontruct_embeddings[i] += mask @ project_matrix @ vh + mean_values_text
            else:
                print("imahe")
                emb_cent = embeddings[i] - mean_values_images
                mask[princ_comp] = (emb_cent @ vh.T)[:, princ_comp].squeeze()
                recontruct_embeddings[i] += ((emb_cent @ vh.T) * mask) @ project_matrix @ vh + mean_values_images

            if return_princ_comp:
                topic_emb_proj_norm = (emb_cent / emb_cent.norm(dim=-1, keepdim=True) @ vh.T) 
                component["cosine_princ_comp"] = torch.abs(topic_emb_proj_norm[:, princ_comp].squeeze()).item() 
                component["correlation_princ_comp"] = (emb_cent @ vh.T)[:, princ_comp].squeeze().item() 
    return recontruct_embeddings, data
