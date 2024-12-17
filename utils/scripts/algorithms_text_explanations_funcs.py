import json
from dataclasses import dataclass
from typing import List
import pandas as pd
from tabulate import tabulate
import torch
import numpy as np
from torchvision.datasets import ImageNet
import random
import matplotlib.pyplot as plt
from utils.misc.visualization import visualization_preprocess

### Layout of data
@dataclass
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

@torch.no_grad()
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

@torch.no_grad()
def get_data_component(data, layer, head, princ_comp):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of dictionaries containing detail for the principal component of interest.
    """

    for entry in data:
        if entry["layer"] == layer and entry["head"] == head and entry["princ_comp"] == princ_comp:
            return [entry]

@torch.no_grad()
def print_data(data, min_princ_comp=-1, is_cosine_present=False):
    """
    Print the collected data in a formatted table.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).
    - is_cosine_present (bool): Whether the correlation values are present in the data.
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
        if is_positive_first:
            positive_texts = texts[:half_length]
            negative_texts = texts[half_length:]
        else:
            positive_texts = texts[half_length:]
            negative_texts = texts[:half_length]
        
        # Pair up corresponding positive and negative texts and collect them in output_rows for tabular display
        for pos, neg in zip(positive_texts, negative_texts):
            pos_text = list(pos.values())[0]
            pos_val  = list(pos.values())[1]
            neg_text = list(neg.values())[0]
            neg_val  = list(neg.values())[1]    
            
            # Append not with maximum value but with the according sign of correlation on the left
            if is_cosine_present:
                corr_sign = np.sign(row.correlation_princ_comp)
                if corr_sign > 0:
                    output_rows.append([pos_text, pos_val, neg_text, neg_val])
                else:
                    output_rows.append([neg_text, neg_val, pos_text, pos_val])

            else:
                output_rows.append([pos_text, pos_val, neg_text, neg_val])
            

        # Print summary information about the current principal component:
        # Including layer, head, principal component index, absolute variance, relative variance, and head rank.
        print(f"Layer {row.layer}, Head {row.head}, Principal Component {row.princ_comp}, "
            f"Variance {row.strength_abs:.3f}, Relative Variance {row.strength_rel:.3f}, Head Rank {row.rank}")
        if is_cosine_present:
            print(f"Cosine of the Topic with the Principal Component {row.cosine_princ_comp * np.sign(row.correlation_princ_comp):.4f}")
        
        # Set the column headers based on whether the first half was considered positive
        if is_positive_first:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]

        # Or set it based on the sign of the correlation
        if is_cosine_present and corr_sign > 0:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]
        
        # Create a DataFrame from the collected rows of positive/negative texts and print it in a formatted table
        output_df = pd.DataFrame(output_rows, columns=columns)
        print(tabulate(output_df, headers='keys', tablefmt='psql'))

@torch.no_grad()
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

@torch.no_grad()
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

@torch.no_grad()
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

@torch.no_grad()
def reconstruct_embeddings(data, embeddings, types, return_princ_comp=False):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings

    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0 or len(types) != len(embeddings):
        assert False, "No embeddings to reconstruct or different lengths."
    # Initialize the reconstructed embeddings
    reconstructed_embeddings = [torch.zeros_like(embeddings[i]) for i in range(len(embeddings))]
    
    # Iterate over each principal component of interest
    for component in data:
        # Retrieve projection matrices and mean values
        project_matrix = torch.tensor(component["project_matrix"])
        vh = torch.tensor(component["vh"])
        princ_comp = component["princ_comp"]

        # Reconstruct Embeddings
        for i in range(len(embeddings)):
            # Derive masking
            mask = torch.zeros((embeddings[i].shape[0], vh.shape[0]))

            if types[i] == "text":
                mask[:, princ_comp] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze()
                reconstructed_embeddings[i] += mask @ project_matrix @ vh 

            else:
                mask[:, princ_comp] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze()
                reconstructed_embeddings[i] += mask @ project_matrix @ vh

            if return_princ_comp:
                topic_emb_proj_norm = (embeddings[i] / embeddings[i].norm(dim=-1, keepdim=True) @ vh.T) 
                component["cosine_princ_comp"] = torch.abs(topic_emb_proj_norm[:, princ_comp].squeeze()).item() 
                component["correlation_princ_comp"] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze().item() 

    return reconstructed_embeddings, data

def reconstruct_top_embedding(data, embedding, type, max_reconstr_score, top_k=10, approx=0.90):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
        - data (list): List of dictionaries containing details for each principal component of interest.
        - embedding: The embedding to reconstruct.
        - type: Type of embedding to reconstruct.
        - max_reconstr_score: Maximum achievable reconstruction score.
        - top_k: Number of top principal components to consider.
        - approx: Approximation threshold for the reconstruction score.

    Returns:
        - data: Data updated with top_k principal components of the given embeddings (if return_princ_comp is True).
    """
    for count, entry in enumerate(data):
        if count == 0:
            # For the first principal component, initialize the query representation
            [query_repres], _ = reconstruct_embeddings([data[count]], [embedding], [type], return_princ_comp=False)

        else:
            # For subsequent components, accumulate their contributions
            [query_repres_tmp], _ = reconstruct_embeddings([data[count]], [embedding], [type], return_princ_comp=False)
            query_repres += query_repres_tmp

        # Normalize the current reconstructed representation
        query_repres_norm = query_repres / query_repres.norm(dim=-1, keepdim=True)
        # Compute the current score: how well this partial reconstruction matches the original embedding
        score = embedding @ query_repres_norm.T

        # If we've reached the top_k limit or our reconstruction score is good enough 
        # (relative to max_reconstr_score and meeting the approx threshold), stop adding more components.
        if count == (top_k - 1) or score / max_reconstr_score > approx:
            top_k = count + 1
            break

    # Print information about how well the reconstruction performed
    percentage = (score / max_reconstr_score * 100).item()
    print(
        f"Reconstruction Quality Report:\n"
        f"- Maximum achievable reconstruction score: {max_reconstr_score.item():.4f}\n"
        f"- Current reconstruction score: {score.item():.4f}\n"
        f"  This corresponds to {percentage:.2f}% of the maximum possible score.\n"
    )

    print(
        f"The reconstruction was performed using the top {top_k} principal component(s).\n "
        f"Increasing this number may improve the reconstruction score.\n\n"
    )

    return data[:top_k]

def image_grid(images, rows, cols, labels=None, scores=None, figsize=None):
    """
    Display a collection of images arranged in a grid with optional labels and scores.
    
    Parameters:
        - images (list): List of image data (e.g., NumPy arrays or PIL Images) to display.
        - rows (int): Number of rows in the image grid.
        - cols (int): Number of columns in the image grid.
        - labels (list, optional): List of labels for each image. Defaults to None.
        - scores (list, optional): List of scores for each image, displayed alongside labels. Defaults to None.
        - figsize (tuple, optional): Size of the figure in inches as (width, height). If not provided, it is calculated based on rows and cols.
    
    Returns:
        - None
    """
    if figsize is None:
        figsize = (cols * 2, rows * 2)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            title = ''
            if labels:
                title += labels[i]
            if scores:
                title += f" ({scores[i]:.4f})" if labels else f"{scores[i]:.4f}"
            if title:
                ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_dataset_imagenet(imagenet_path, transform, samples_per_class=3, tot_samples_per_class=50, seed=50):
    """
    Create a balanced subset of the ImageNet validation dataset for nearest-neighbor (NN) search.

    Args:
    - imagenet_path (str): Path to the root directory of the ImageNet dataset.
    - transform (callable): A function/transform to apply to the images in the dataset.
    - samples_per_class (int, optional): Number of samples to pick per class. Default is 3.
    - tot_samples_per_class (int, optional): Total number of samples per class considered as a block. Default is 50.
    - seed (int, optional): Random seed for reproducibility. Default is 50.

    Returns:
    - torch.utils.data.Subset: A subset of the ImageNet validation dataset containing the selected samples.
    """
    # Load the ImageNet validation dataset
    ds_vis = ImageNet(root=imagenet_path, split="val", transform=transform)

    # all_indices: a list of indices representing every sample in the validation set.
    all_indices = list(range(len(ds_vis)))
    nr_classes = len(ds_vis.classes)  # number of classes in the dataset

    # Set the random seed for reproducibility
    random.seed(seed)

    # For each class, randomly pick 'samples_per_class' samples from a block of 'tot_samples_per_class' samples.
    # This results in a balanced subset of classes.
    index = [random.sample(all_indices[x*tot_samples_per_class:(x + 1)*tot_samples_per_class], samples_per_class) 
             for x in range(nr_classes)]

    # Flatten the list of indices (since it's currently a list of lists)
    index = [x for xs in index for x in xs]

    # Create subsets of the dataset and visualization dataset using the selected indices
    ds_vis = torch.utils.data.Subset(ds_vis, index)
    return ds_vis


def create_dbs(scores_array_images, scores_array_texts, nr_top_imgs=20, nr_worst_imgs=20, nr_cont_imgs=20, text_query = None, max_reconstr_score = -1):
    """
    Create three sets of data containing top, worst, and continuous cosine similarity results 
    for images and corresponding text queries.

    Args:
    - data (iterable): The source dataset containing image and text samples.
    - scores_array_images (np.ndarray): Array of image scores, structured with 'score' and index values.
    - scores_array_texts (np.ndarray): Array of text scores, structured with 'score' and index values.
    - nr_top_imgs (int, optional): Number of top (highest cosine similarity) images to retrieve. Default is 20.
    - nr_worst_imgs (int, optional): Number of worst (lowest cosine similarity) images to retrieve. Default is 20.
    - nr_cont_imgs (int, optional): Number of continuous (interpolated similarity) images to retrieve. Default is 20.
    - text_query (str): The original query text for which cosine similarities are analyzed.
    - max_reconstr_score (float or torch.Tensor): The maximum reconstruction strength to display alongside results.

    Returns:
    - list of tuples: Each tuple contains:
        - db_images (np.ndarray): Selected image data based on scores.
        - db_texts (np.ndarray): Selected text data based on scores.
        - description (str): A string describing the type of retrieval (top, worst, or continuous).
        - num_samples (int): Number of samples in the current retrieval.
    """
    # Sort indices of scores
    sorted_scores_images = np.sort(scores_array_images, order='score')
    sorted_scores_texts = np.sort(scores_array_texts, order='score')

    # Top scores
    top_dbs_images = sorted_scores_images[-nr_top_imgs:][::-1]  # Get top `nr_top_imgs` elements, sorted descending
    top_dbs_texts = sorted_scores_texts[-nr_top_imgs:][::-1]  # Get top `nr_top_imgs` elements, sorted descending

    # Worst scores
    worst_dbs_images = sorted_scores_images[:nr_worst_imgs]  # Get worst `nr_worst_imgs` elements, sorted ascending
    worst_dbs_texts = sorted_scores_texts[:nr_worst_imgs]  # Get worst `nr_worst_imgs` elements, sorted ascending

    # Continuous scores (step-based interpolation)
    t = np.linspace(0, 1, num=nr_cont_imgs)  # Uniformly spaced in [0, 1]
    scaled_t = 0.5 * (1 - np.cos(np.pi * t))  # Cosine transformation for symmetry
    nonlinear_indices_images = (scaled_t * (len(sorted_scores_images) - 1)).astype(int)
    cont_dbs_images = sorted_scores_images[nonlinear_indices_images]
    nonlinear_indices_texts = (scaled_t * (len(sorted_scores_texts) - 1)).astype(int)
    cont_dbs_texts = sorted_scores_texts[nonlinear_indices_texts]

    # Prepare three sets of results to display:
    # 1. Highest cosine similarity samples
    # 2. Most negative cosine similarity samples
    # 3. A continuous range of samples with intermediate similarity
    dbs = [
        (top_dbs_images, top_dbs_texts, 
         f"Printing highest cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if 
         text_query is not None else f"Printing highest cosine similarity",
         nr_top_imgs),
        (worst_dbs_images, worst_dbs_texts, 
         f"Printing most negative cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if 
         text_query is not None else f"Printing most negative cosine similarity",
         nr_worst_imgs),
        (cont_dbs_images, cont_dbs_texts, 
         f"Printing continuous cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if
         text_query is not None else f"Printing continuous cosine similarity",
         nr_cont_imgs)
    ]

    return dbs


def visualize_dbs(data, dbs, ds_vis, texts_str, imagenet_classes, text_query= None):
    """
    Visualize and analyze subsets of images and their associated text contributions 
    based on cosine similarity scores.

    Args:
    - data (iterable): Data containing principal components or details to analyze.
    - dbs (list of tuples): List of datasets to visualize, where each tuple contains:
        - db (list of tuples): Subset of images with scores and indices.
        - db_text (list of tuples): Subset of text entries with scores and indices.
        - display_text (str): Header text describing the current subset (e.g., "Top scores").
        - length (int): Number of samples in the current subset.
    - ds_vis (torch.utils.data.Dataset): Image dataset (e.g., ImageNet validation dataset).
    - texts_str (list of str): List of textual descriptions or queries corresponding to text indices.
    - text_query (str): The original query text being analyzed.
    - imagenet_classes (list of str): List of class names in the ImageNet dataset.

    Returns:
    - None
    """
    # Print a header to indicate which text query we are analyzing
    if text_query is not None:
        print(f"----Decomposition for text: '{text_query}'----")
        print_data(data, is_cosine_present=True)

    else:
        print(f"----Decomposition of Principal Component----")
        print_data(data, is_cosine_present=False)


    for db, db_text, display_text, length in dbs:
        images, labels, scores = [], [], []

        # Collect image samples, their class labels, and similarity scores
        for score, image_index in db:
            images.append(ds_vis[image_index][0])  # Image data
            labels.append(imagenet_classes[ds_vis[image_index][1]])  # Class label
            scores.append(score)  # Cosine similarity score

        # Print the header describing the current subset
        print(display_text)

        # Prepare and display the corresponding texts with their scores
        output_rows = []
        for score, text_index in db_text:
            output_rows.append([texts_str[text_index], score])
        output_df = pd.DataFrame(output_rows, columns=["Text", "Strength"])
        print(tabulate(output_df, headers='keys', tablefmt='psql'))

        # Display the images in a grid layout
        rows, cols = (length // 4, 4)  # Grid layout: 4 columns, rows derived from the number of images
        image_grid(images, rows, cols, labels=labels, scores=scores)

def visualize_principal_component(
    layer, head, princ_comp, nr_top_imgs, nr_worst_imgs, nr_cont_imgs,
    attention_dataset, final_embeddings_images, final_embeddings_texts, 
    seed, imagenet_path, texts_str, imagenet_classes, 
    transform=visualization_preprocess
):
    """
    Visualize the top, worst, and continuous cosine similarity scores for images and texts 
    corresponding to a specific principal component of interest.

    Args:
    - layer (int): The attention layer to analyze.
    - head (int): The attention head within the specified layer.
    - princ_comp (int): The principal component index to visualize.
    - nr_top_imgs (int): Number of top (highest similarity) images to retrieve and display.
    - nr_worst_imgs (int): Number of worst (lowest similarity) images to retrieve and display.
    - nr_cont_imgs (int): Number of continuous (interpolated similarity) images to retrieve and display.
    - attention_dataset (str): Path to the JSON file containing attention data.
    - final_embeddings_images (torch.Tensor): Final image embeddings for similarity computation.
    - final_embeddings_texts (torch.Tensor): Final text embeddings for similarity computation.
    - seed (int): Random seed for reproducibility when creating the dataset.
    - imagenet_path (str): Path to the root directory of the ImageNet dataset.
    - texts_str (list of str): List of text descriptions corresponding to text embeddings.
    - imagenet_classes (list of str): List of class names in the ImageNet dataset.
    - transform (callable, optional): Transformation function to preprocess ImageNet images. 
                                      Defaults to `visualization_preprocess`.

    Returns:
    - None
    """
    # Retrieve and filter the principal component data
    data = get_data(attention_dataset, -1, skip_final=True)
    data = get_data_component(data, layer, head, princ_comp)

    # Load a subset of the ImageNet dataset
    ds_vis = create_dataset_imagenet(
        imagenet_path, transform, samples_per_class=3, 
        tot_samples_per_class=50, seed=seed
    )

    # Initialize arrays to store similarity scores for images and texts
    scores_array_images = np.empty(
        final_embeddings_images.shape[0], 
        dtype=[('score', 'f4'), ('img_index', 'i4')]
    )
    scores_array_texts = np.empty(
        final_embeddings_texts.shape[0], 
        dtype=[('score', 'f4'), ('txt_index', 'i4')]
    )

    # Create arrays of indices for referencing images and texts
    indexes_images = np.arange(0, final_embeddings_images.shape[0], 1)
    indexes_texts = np.arange(0, final_embeddings_texts.shape[0], 1)

    # Compute mean embeddings for centering
    mean_final_images = torch.mean(final_embeddings_images, axis=0)
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    images_centered = final_embeddings_images - mean_final_images
    texts_centered = final_embeddings_texts - mean_final_texts

    # Normalize embeddings to unit norm
    texts_centered /= texts_centered.norm(dim=-1, keepdim=True)
    images_centered /= images_centered.norm(dim=-1, keepdim=True)

    # Compute cosine similarity scores with the specified principal component
    vh = torch.tensor(data[0]["vh"])
    scores_array_images["score"] = ((images_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_images["img_index"] = indexes_images

    scores_array_texts["score"] = ((texts_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_texts["txt_index"] = indexes_texts

    # Create datasets for visualization
    dbs = create_dbs(
        scores_array_images, scores_array_texts, nr_top_imgs, 
        nr_worst_imgs, nr_cont_imgs, text_query=None, max_reconstr_score=-1
    )

    # Visualize the results
    visualize_dbs(data, dbs, ds_vis, texts_str, imagenet_classes, text_query=None)
