import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(420)
np.random.seed(420)
@torch.no_grad()

def text_span(data, text_features, texts, layer, head, text_per_princ_comp, device, iters=80, rank=80):
    """
    This function performs iterative removal and reconstruction of the attention head matrix
    using the provided text features.

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        text_per_princ_comp: Number of texts to consider for each princ_comp.
        device: The device to perform computations on.
        iters=80: Number of iteration of the algorithm (i.e. number of texts per head)
        rank=80: Rank on which to prune the activation space.

    Returns:
        reconstruct: The reconstructed attention head matrix.
        results: List of text descriptions with maximum variance.
    """
    results = []
    # Svd of attention head matrix
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    text_features = text_features - mean_values_text
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

    results = [{"text": text} for text in results]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object


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
    
    # Total strength eigenvectors
    indexes = torch.cat((indexes_max, indexes_min))
    strength = torch.cat((s, s), dim=0)
 

    return indexes, strength

def solve_non_negative_least_squares(D, X, num_iter=10000, lr=1e-3):
    """
    Solves for A in D ≈ A * X with A ≥ 0 using PyTorch.

    Parameters:
    D (torch.Tensor): Target matrix of shape (m, n).
    X (torch.Tensor): Input matrix of shape (p, n).
    num_iter (int): Number of optimization iterations.
    lr (float): Learning rate for the optimizer.

    Returns:
    torch.Tensor: Solution matrix A of shape (m, p) with non-negative entries.
    """
    # Ensure X and D are float tensors and on the same device
    X = X.float()
    D = D.float()
    device = X.device
    D = D.to(device)
    
    m, n = D.shape
    p = X.shape[0]
    
    # Initialize A with random values
    A_param = torch.randn(m, p, device=device, requires_grad=True)
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam([A_param], lr=lr)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        # Enforce non-negativity using softplus
        AX = torch.matmul(A_param, X)
        # Compute the loss (Frobenius norm squared)
        loss = torch.norm(D - AX, p='fro')**2
        loss.backward()
        optimizer.step()

        # Clip paramteres to keep them positive
        A_param.data.clamp_(0)

    print(loss)
    return A_param

def splice_data_approx(data, text_features, texts, layer, head, seed, dataset, iters, rank, device):
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
    writer = SummaryWriter("logs_test")
    print(f"\nLayer [{layer}], Head: {head}")
    # Define tag prefixes for this layer, head, and seed
    tag_prefix = f"Dataset_{dataset}/Layer_{layer}/Head_{head}/Seed_{seed}"
    
    # Center text and image data
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    
    # Project text_features to data lower-rank eigenspace with required rank
    u , s, vh = torch.linalg.svd(data, full_matrices=False)
    # Write rank of matrix
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine rank where cumulative variance exceeds 99% of total variance
    threshold = 0.99
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1

    vh = vh[:rank]
    text_features = text_features @ vh.T @ vh
    text_features = text_features / torch.linalg.norm(text_features, axis=-1)[:, np.newaxis]
    simil_matrix = (text_features @ text_features.T) # Nxd * dxM, cos similarity on each row
    # Initialize A with required gradient and with initial range guess
    scale = data.max()
    A = torch.clamp(scale*torch.rand(data.shape[0], text_features.shape[0], requires_grad=True, device=device) + \
                    2*scale, 2*scale, 3*scale)
    A_ = A.clone().detach().requires_grad_(True)

    # Set up optimizer and parameters
    optimizer = torch.optim.Adam([A_], lr=0.001)
    epochs_main = 20000
    
    # Initial ratio bewteen regularization and rmse loss 
    lbd_l1 = 1
    ratio = 1

    ## First part: main optimization loop
    patience = 500  # Number of epochs to wait for improvementy
    # Initialize variables for early stopping
    prev_cum_sum = None
    prev_indexes = torch.tensor([x for x in range(iters)], device=device)
    prev_relative_strength = None
    prev_loss = None
    stabilization_window = 50  # Number of iterations to check stability
    cum_sum_stable_count = 0  # Counter for indexes change stability
    relative_strength_stable_count = 0  # Counter for relative strength stability
    stabilization_threshold_cum = int(min(iters, rank)) + 1 # Percentage

    stabilization_threshold_strength = 0.01    
    # Training loop with early stopping
    for epoch in range(epochs_main):
        optimizer.zero_grad()  # Clear gradients from previous step

        # Clip paramteres to keep them positive
        A = torch.nn.functional.gelu(A_)
        # Compute the product A @ text_features using only stronger "iters" text with highest std across data
        text_features_std = A.std(axis=0)
        indexes = torch.argsort(text_features_std, descending=True)[:iters]
        pred = A[:, indexes] @ text_features[indexes, :]

        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred-data)**2))
        # Regularization L1 on row *used* for predictions (i.e. sparse row i.e. fewer text embeddings)
        # and L_inf on *used* for predictions columns
        loss_l1 = ratio * lbd_l1 * (torch.norm(A[:, indexes], p=1, dim=1).mean() + \
                            torch.norm(A[:, indexes], p=float('inf'), dim=0).mean())

        loss = loss_l1 + loss_rmse

        # Use a lbd_1 of 1:1 of loss functions
        if epoch == 0:
            lbd_l1 = ratio * lbd_l1 * loss_rmse.detach().clone()/loss_l1.detach().clone()
            continue
        
        # Backpropagation
        loss.backward()
        
        # Update A using the optimizer
        optimizer.step()

        # Compute metrics for early stopping
        text_str = text_features_std[indexes]
        tot_str = torch.sum(text_str)
        all_str = torch.sum(text_features_std)
        relative_strength = 100 * tot_str / all_str
        cum_sum = (prev_indexes[:stabilization_threshold_cum] != indexes[:stabilization_threshold_cum]).sum()

        # Log to TensorBoard every patience epochs
        if epoch % patience == 0:
            # Log the loss to TensorBoard
            writer.add_scalar(f"{tag_prefix}/Loss/RMSE", loss_rmse, epoch)
            writer.add_scalar(f"{tag_prefix}/Loss/L1", loss_l1, epoch)
            writer.add_scalar(f"{tag_prefix}/Loss/loss", loss, epoch)
 
            # Log additional metadata (e.g., loss or total strength)
            writer.add_scalar(f"{tag_prefix}/relative_strength", relative_strength, epoch)

            writer.add_scalar(f"{tag_prefix}/indexes_cum_sum", cum_sum, epoch)

            writer.add_histogram(f"{tag_prefix}/indexes", indexes, epoch)

            writer.add_histogram(f"{tag_prefix}/top_std_features_rel", 100*text_features_std[indexes][:iters]/tot_str, epoch)

        # Check stability for `cum_sum`
        if prev_cum_sum is not None:
            cum_sum_change = cum_sum 
            if cum_sum_change <= 1:
                cum_sum_stable_count += 1
            else:
                cum_sum_stable_count = max(0, cum_sum_stable_count - 1)   # Reset if not stable

        # Check stability for `relative_strength`
        if prev_relative_strength is not None:
            relative_strength_change = abs((relative_strength - prev_relative_strength) / prev_relative_strength * 100)
            if relative_strength_change < stabilization_threshold_strength:
                relative_strength_stable_count += 1
            else:
                relative_strength_stable_count = 0  # Reset if not stable

        # Update previous values
        prev_cum_sum = cum_sum
        prev_indexes = indexes
        prev_relative_strength = relative_strength

        # Check if both metrics are stable for the required window
        if cum_sum_stable_count >= stabilization_window and relative_strength_stable_count >= stabilization_window:
            print(f"Early stopping at epoch {epoch} due to stabilization.")
            break

    # Log text features
    text_str = text_features_std[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_std)   
    results = [
        {
            "text": texts[idx],
            "strength_abs": text_str[i].item(),
            "strength_rel": (100 * text_str[i] / tot_str).item(),
        }
        for i, idx in enumerate(indexes)
    ]
    # Generate Markdown-formatted table
    markdown_table = "| Text         | Absolute Strength | Relative Strength (%) |\n"
    markdown_table += "|--------------|-------------------|------------------------|\n"
    for result in results:
        markdown_table += f"| {result['text']} | {result['strength_abs']:.4f}         | {result['strength_rel']:.2f}                   |\n"
    # Log LaTeX table as text (enable slider with epoch step)
    writer.add_text(f"{tag_prefix}/Top-K Strengths for Epoch", markdown_table, epoch)   

    # Take columns of A with highest std (i.e. more active columns -> more active text embedding)
    text_features_std = A.std(axis=0)
    indexes = torch.argsort(text_features_std, descending=True)[:iters]
    A = A[:, indexes].detach().clone().requires_grad_(True)
    text_features = text_features[indexes, :].detach().clone().requires_grad_(True)

    ## Second part: finetune over rmse loss
    # Training loop with early stopping
    patience_counter = 0
    patience = 100  # Number of epochs to wait for improvement
    min_delta = 1e-9  # Minimum improvement in loss to be considered
    A = A.requires_grad_(True) 
    A.data.clamp_(0)

    optimizer = torch.optim.Adam([A], lr=0.001)  # Recreate the optimizer
    epochs = 500
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Clear gradients from previous step
        optimizer.zero_grad()  
        # Make prediction
        pred = A @ text_features
        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred-data)**2))
        loss = loss_rmse
        # Backpropagation
        loss.backward()
        # Update A using the optimizer
        optimizer.step()
        # Clip parameteres to keep them positive
        A.data.clamp_(0)

        # Log values
        writer.add_scalar(f"{tag_prefix}/Loss/RMSE", loss_rmse, epochs_main + epoch) 
        writer.add_histogram(f"{tag_prefix}/indexes", indexes, epochs_main + epoch)

        # Early stopping logic
        if loss < best_loss - min_delta:
            best_loss = loss_rmse
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1  # Increment if no improvement

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} with best loss_rmse: {best_loss.item():.6f}")
            break

    # Log second time  
    text_str = text_features_std[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_std)   
    results = [
        {
            "text": texts[idx],
            "strength_abs": text_str[i].item(),
            "strength_rel": (100 * text_str[i] / tot_str).item(),
        }
        for i, idx in enumerate(indexes)
    ]
    # Generate Markdown-formatted table
    markdown_table = "| Text         | Absolute Strength | Relative Strength (%) |\n"
    markdown_table += "|--------------|-------------------|------------------------|\n"
    for result in results:
        markdown_table += f"| {result['text']} | {result['strength_abs']:.4f}         | {result['strength_rel']:.2f}                   |\n"
    # Log LaTeX table as text (enable slider with epoch step)
    writer.add_text(f"{tag_prefix}/Top-K Strengths for Epoch", markdown_table, epochs_main + epoch)   

    writer.close()

    # Retrieve corresponding text
    text_str = text_features_std[indexes].cpu()
    tot_str = torch.sum(text_str).cpu() # Total strength of text embeddings on that
    results = [{"text": texts[idx], "strength_abs": text_str[i].item(), "strength_rel": (100 * text_str[i] / tot_str).item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Compute reconstruction using only our selected 
    A = A.clamp(0, None)
    reconstruct = A.detach().cpu().numpy() @ text_features.detach().cpu().numpy()
    
    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object

    
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
    mean_values_text = np.mean(text_features, axis=0)
    text_features = text_features - mean_values_text
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
    lmbda = 0.1 # Regularisation weight to make matrix denser
    n_epochs = 500 # Number of epochs
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
        
    reconstruct = U @ V.T

    # Get total strength of text embedding basis as an average
    text_str = np.mean(U,axis=0) # Strength of a text embedding across its contributions
    tot_str = np.sum(text_str) # Total strength of text embeddings
    sort = np.argsort(text_str)[::-1]
    text_str = text_str[sort]
    indexes = indexes[sort]    
    results = [{"text": texts[idx], "strength_abs": text_str[i].astype(float), "strength_rel": (100 * text_str[i] / tot_str).astype(float)} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct + mean_values_att, json_object


def spih_data_approx(data, text_features, texts, layer, head, seed, dataset, device):
    """
    This function finds a sparse (nr_basis_elem) non-negative approximation of the attention head matrix
    using the provided text features. (i.e. A @ text_features = data s.t. A > 0  and A sparse)

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        seed: The current seed of the text dataset.
        dataset": The current text dataset used.
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix using the found basis.
        results: Jsonline file containing the found basis and metadata.
    """

    # Setup Writer Tensorboard
    writer = SummaryWriter("logs")
    print(f"\nLayer [{layer}], Head: {head}")

    # Define tag prefixes for this layer, head, and seed
    tag_prefix = f"Dataset_{dataset}/Layer_{layer}/Head_{head}/Seed_{seed}"
    
    # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=True)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top eigenvectors of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]

    nr_basis_elem = 2*rank
    print(nr_basis_elem)
    # Project text_features to data lower-rank eigenspace (removes redundant informations)
    text_features = text_features @ vh.T
    # Get initialization parameters for the coefficient matrix of A
    # indexes, strength = svd_parameters_init(vh, s, text_features_mem, rank)
    
    # Initialize A with required gradient and with initial range guess
    # strength = strength - strength.min()/(strength.max() - strength.min()) # min-max normalization
    # Cosine similarity
    data_rec = u @ torch.diag_embed(s)
    data_rec = data_rec / torch.linalg.norm(data_rec, axis=-1)[:, np.newaxis]
    # Normalize text features
    text_features_mem = text_features / torch.linalg.norm(text_features, axis=-1)[:, np.newaxis]
    sim_matrix_text = text_features_mem @ text_features_mem.T
    A = data_rec @ text_features_mem.T #torch.empty(data.shape[0], text_features.shape[0], device=device)
    # Bring back on order of prediction
    A = 2*torch.max(data, dim=-1).values.unsqueeze(-1)*A
    # Set initialization parameters for text embedding with given strength (same initial std and mean per embedding)
    #A[:, indexes] = data.max()* #(strength.unsqueeze(0)*torch.rand(data.shape[0], indexes.shape[0],  device=device)) + data.max()
    #mask = torch.ones(A.shape[1], dtype=bool, device=device)
    #mask[indexes] = False 
    # Set all the other parameters to min strength (same initial std and mean per embedding)
    #A[:, mask] = data.max()* #strength.min()*torch.rand(data.shape[0], mask.shape[0] - torch.unique(indexes).shape[0],  device=device) + data.max()
    A = A.clone().detach().requires_grad_(True)
    # Clip paramteres to keep them positive
    A.data.clamp_(min=0)
    # Set up optimizer and parameters
    optimizer = torch.optim.Adam([A], lr=0.001)
    epochs_main = 10000
    
    # Initial ratio bewteen regularization l1 and rmse loss 
    lbd_l1 = 1
    ratio = 1

    ## First part: main optimization loop
    # Initialize variables for early stopping
    prev_cum_sum = None
    prev_indexes = torch.tensor([x for x in range(nr_basis_elem)], device=device)
    prev_relative_strength = None
    stabilization_window = 500  # Number of iterations to check stability
    cum_sum_stable_count = 0  # Counter for indexes change stability
    relative_strength_stable_count = 0  # Counter for relative strength stability
    stabilization_threshold_cum = int(min(nr_basis_elem, rank)) + 1 # Percentage
    stabilization_threshold_strength = 0.01    
    patience = 500  # Number of epochs to log something

    # Training loop with early stopping
    for epoch in range(epochs_main):
        
        optimizer.zero_grad()  # Clear gradients from previous step

        # Compute the product A @ text_features using only stronger "nr_basis_elem" text with highest std and mean across data
        text_features_strength = A.mean(axis=0)
        indexes = torch.argsort(text_features_strength, descending=True)[:nr_basis_elem]
        pred = A[:, indexes] @ text_features[indexes, :]

        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred @ vh -data)**2))
        # Regularization L1 on row *used* for predictions (i.e. sparse row i.e. fewer text embeddings)
        # and L_inf on *used* for predictions columns
        loss_l1 = ratio * lbd_l1 * (torch.norm(A[:, indexes], p=1, dim=1).mean() + \
                            torch.norm(A[:, indexes], p=float('inf'), dim=0).mean() + \
                            sim_matrix_text[indexes, indexes].mean())

        loss = loss_l1 + loss_rmse

        # Use a lbd_1 of 1:1 of loss functions (init at first iteration)
        if epoch == 0:
            lbd_l1 = ratio * lbd_l1 * loss_rmse.detach().clone()/loss_l1.detach().clone()
            epoch += 1
            continue
        
        # Backpropagation
        loss.backward()
        
        # Update A using the optimizer
        optimizer.step()

        # Compute metrics for early stopping
        text_str = text_features_strength[indexes]
        tot_str = torch.sum(text_str) + 1e-9
        all_str = torch.sum(text_features_strength) + 1e-9

        relative_strength = 100 * tot_str / all_str

        # Calculate the cumulative sum
        cum_sum = (prev_indexes[:stabilization_threshold_cum] != indexes[:stabilization_threshold_cum]).sum()

        # Log to TensorBoard every patience epochs
        if epoch % patience == 0:
            # Log the loss to TensorBoard
            writer.add_scalar(f"{tag_prefix}/Loss/RMSE", loss_rmse, epoch)
            writer.add_scalar(f"{tag_prefix}/Loss/L1", loss_l1, epoch)
            writer.add_scalar(f"{tag_prefix}/Loss/loss", loss, epoch)
 
            # Log additional metadata (e.g., loss or total strength)
            writer.add_scalar(f"{tag_prefix}/relative_strength", relative_strength, epoch)

            writer.add_scalar(f"{tag_prefix}/indexes_cum_sum", cum_sum, epoch)

        # Check stability for `relative_strength`
        if prev_relative_strength is not None:
            relative_strength_change = abs(relative_strength - prev_relative_strength)
            if relative_strength_change > 99.9:
                relative_strength_stable_count += 1
            else:
                relative_strength_stable_count = 0  # Reset if not stable
        # Check stability for `cum_sum`
        if prev_cum_sum is not None:
            cum_sum_change = abs(cum_sum - prev_cum_sum)
            if cum_sum_change < 1:
                cum_sum_stable_count += 1
            else:
                cum_sum_stable_count = 0  # Reset if not stable

        if relative_strength_stable_count > stabilization_window and cum_sum_stable_count > stabilization_window:
            print(f"Early stopping at epoch {epoch + 1} with best loss_rmse: {best_loss.item():.6f}")
            break

        # Update previous values
        prev_relative_strength = relative_strength
        prev_cum_sum = cum_sum
        prev_indexes = indexes

        # Clip paramteres to keep them positive
        A.data.clamp_(min=0)

    ## Log found text features
    text_str = text_features_strength[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_strength)   
    results = [
        {
            "text": texts[idx],
            "strength_abs": text_str[i].item(),
            "strength_rel": (100 * text_str[i] / tot_str).item(),
        }
        for i, idx in enumerate(indexes)
    ]
    # Generate Markdown-formatted table
    markdown_table = "| Text         | Absolute Strength | Relative Strength (%) |\n"
    markdown_table += "|--------------|-------------------|------------------------|\n"
    for result in results:
        markdown_table += f"| {result['text']} | {result['strength_abs']:.4f}         | {result['strength_rel']:.2f}                   |\n"
    # Log LaTeX table as text (enable slider with epoch step)
    writer.add_text(f"{tag_prefix}/Top-K Strengths for Epoch", markdown_table, epoch)   


    # Take columns of A with highest mean and std (i.e. more active columns -> more active text embedding)
    text_features_strength = A.mean(axis=0)
    indexes = torch.argsort(text_features_strength, descending=True)[:nr_basis_elem]
    A = A[:, indexes].detach().clone().requires_grad_(True)
    text_features = text_features[indexes, :].detach().clone().requires_grad_(True)

    ## Second part: finetune over rmse loss
    # Training loop with early stopping
    patience_counter = 0
    patience = 100  # Number of epochs to wait for improvement
    min_delta = 1e-9  # Minimum improvement in loss to be considered
    A = A.requires_grad_(True) 
    A.data.clamp_(0)

    optimizer = torch.optim.Adam([A], lr=0.001)  # Recreate the optimizer
    epochs = 500
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Clear gradients from previous step
        optimizer.zero_grad()  

        # Make prediction
        pred = A @ text_features
        # Compute the sqrt mean squared error loss
        loss_rmse = torch.sqrt(torch.mean((pred @ vh-data)**2))
        loss = loss_rmse
        # Backpropagation
        loss.backward()
        # Update A using the optimizer
        optimizer.step()

        # Log values
        writer.add_scalar(f"{tag_prefix}/Loss/RMSE", loss_rmse, epochs_main + epoch) 
        writer.add_histogram(f"{tag_prefix}/indexes", indexes, epochs_main + epoch)

        # Early stopping logic
        if loss < best_loss - min_delta:
            best_loss = loss_rmse
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1  # Increment if no improvement

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} with best loss_rmse: {best_loss.item():.6f}")
            break

        # Clip paramteres to keep them positive
        A.data.clamp_(min=0)

    # Log second time  
    text_str = text_features_strength[indexes]
    tot_str = torch.sum(text_str)
    all_str = torch.sum(text_features_strength)   
    results = [
        {
            "text": texts[idx],
            "strength_abs": text_str[i].item(),
            "strength_rel": (100 * text_str[i] / tot_str).item(),
        }
        for i, idx in enumerate(indexes)
    ]
    # Generate Markdown-formatted table
    markdown_table = "| Text         | Absolute Strength | Relative Strength (%) |\n"
    markdown_table += "|--------------|-------------------|------------------------|\n"
    for result in results:
        markdown_table += f"| {result['text']} | {result['strength_abs']:.4f}         | {result['strength_rel']:.2f}                   |\n"
    # Log LaTeX table as text (enable slider with epoch step)
    writer.add_text(f"{tag_prefix}/Top-K Strengths for Epoch", markdown_table, epochs_main + epoch)   

    writer.close()

    # Retrieve corresponding text
    text_str = text_features_strength[indexes].cpu()
    tot_str = torch.sum(text_str).cpu() # Total strength of text embeddings on that
    results = [{"text": texts[idx], "strength_abs": text_str[i].item(), "strength_rel": (100 * text_str[i] / tot_str).item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Compute reconstruction using only our selected 
    A = A.clamp(0, None)
    reconstruct = A.detach().cpu().numpy() @ text_features.detach().cpu().numpy()
    
    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }
    return reconstruct @ vh.detach().cpu().numpy() + mean_values_att, json_object

    


def spih_data_approx_det(data, text_features, texts, layer, head, seed, dataset, nr_basis_elem, device):
    """
    This function finds a sparse (nr_basis_elem) non-negative approximation of the attention head matrix
    using the provided text features. (i.e. A @ text_features = data s.t. A > 0  and A sparse)

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        seed: The current seed of the text dataset.
        dataset": The current text dataset used.
        nr_basis_elem: Number of iterations to perform.
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix using the found basis.
        results: Jsonline file containing the found basis and metadata.
    """
    print(f"\nLayer [{layer}], Head: {head}")

    # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=True)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top eigenvectors of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]
    nr_basis_elem = 2*rank

    # Project text_features to data lower-rank eigenspace (removes redundant informations)
    text_features = text_features @ vh.T
    # Get initialization parameters for the coefficient matrix of A
    
    # Initialize A with required gradient and with initial range guess
    data_rec = u @ torch.diag_embed(s)

    total_variance = torch.trace(data_rec.T @ data_rec)
    # Do not normalize text features
    sim_matrix_text = text_features @ text_features.T

    # Bring back on order of prediction
    # A = 2*torch.max(data, dim=-1).values.unsqueeze(-1)*A

    reconstruct = torch.zeros_like(data_rec)

    indexes = torch.empty(nr_basis_elem, device=device)
    strength = torch.empty(nr_basis_elem + 1, device=device)
    strength[0] = total_variance
    strength_diff = 0
    # Reconstruct attention head matrix by using projection on nr. iters max variance texts embeddings
    for i in range(nr_basis_elem):

        # Projects each data point (rows in data) into the feature space defined by the text embeddings.
        # Each row in projection now represents how each attention head activation vector i aligns with each text embedding j (i, j),
        # quantifying the contribution of each text_feature to the data in this iteration.

        cosine_similarity = (data_rec ) @ (text_features).T # Nxd * dxM, cos similarity on each row 

        projection_mean = cosine_similarity.mean(axis=0)
    
        # Take top text embedding with max variance for the data matrix
        top_n = torch.argmax(projection_mean)
        # Save index of text embedding
        indexes[i] = top_n

        # Remove contribution from projection only if cosine is positive
        cosine_similarity_top_n = cosine_similarity[:, top_n]
        positive_mask = cosine_similarity_top_n > 0
        # Rank 1 approximation 
        text_norm = text_features[top_n] @ text_features[top_n].T
        rank_1_approx = torch.zeros_like(data_rec)
        rank_1_approx[positive_mask] = (data_rec[positive_mask] @ text_features[top_n] / text_norm)[:, np.newaxis] \
                        * text_features[top_n][np.newaxis, :]
        reconstruct += rank_1_approx

        # Remove contribution from data matrix
        data_rec = data_rec - rank_1_approx


        strength[i+1] = torch.trace(data_rec.T @ data_rec)
        print(strength[i+1].item())
    
    results = [{"text": texts[int(idx.item())], "strength_abs": (strength[i] - strength[i+1]).item(), "strength_rel": (100 * (strength[i] - strength[i+1]) / total_variance).item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }

    for result in results:
        print(result)


    return reconstruct @ vh + mean_values_att, json_object

def highest_cos_sim_head(data, text_features, texts, layer, head, seed, dataset, nr_basis_elem, device):
    """
    This function finds a sparse (nr_basis_elem) non-negative approximation of the attention head matrix
    using the provided text features. (i.e. A @ text_features = data s.t. A > 0  and A sparse)

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        seed: The current seed of the text dataset.
        dataset": The current text dataset used.
        nr_basis_elem: Number of iterations to perform.
        device: The device to perform computations on.

    Returns:
        reconstruct: The reconstructed attention head matrix using the found basis.
        results: Jsonline file containing the found basis and metadata.
    """
    print(f"\nLayer [{layer}], Head: {head}")

    # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=True)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top eigenvectors of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]
    nr_basis_elem = 2*rank

    # Project text_features to data lower-rank eigenspace (removes redundant informations)
    text_features = text_features @ vh.T
    # Get initialization parameters for the coefficient matrix of A
    
    # Initialize A with required gradient and with initial range guess
    # Cosine similarity
    data_rec = u @ torch.diag_embed(s)
    data_rec = data_rec

    # Normalize text features
    # text_features = text_features / torch.linalg.norm(text_features, axis=-1)[:, np.newaxis]

    reconstruct = torch.zeros_like(data_rec)

   
    cosine_similarity = (data_rec ) @ (text_features ).T # Nxd * dxM, cos similarity on each row 

    projection_mean = cosine_similarity.mean(axis=0)

    indexes = torch.argsort(projection_mean, descending=True)[:nr_basis_elem]

    results = [{"text": texts[int(idx.item())], "strength_abs": projection_mean[i].item()} for i, idx in enumerate(indexes)]    # Reconstruct original matrix with new basis

    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": vh.tolist(),
        "embeddings_sort": results
    }

    for result in results:
        print(result)


    return reconstruct @ vh + mean_values_att, json_object
