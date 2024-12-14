import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import torch


def load_embeddings_batch(store, non_error_index, batch_size=1000):
    """
    Load embeddings in batches and yield them
    """
    for i in range(0, len(non_error_index), batch_size):
        batch_indices = non_error_index[i : i + batch_size]
        batch_embeddings = []

        for idx in batch_indices:
            emb = store[idx]
            if isinstance(emb, torch.Tensor):
                emb = emb.numpy()
            batch_embeddings.append(emb.reshape(-1))

        yield np.array(batch_embeddings)


def compute_principal_components(store, non_error_index, batch_size=1000):
    """
    Compute all 300 principal components of the embedding space using IncrementalPCA
    """
    # Initialize IncrementalPCA with full dimensionality
    ipca = IncrementalPCA(n_components=300)  # Get all 300 components

    # Fit PCA incrementally
    for batch in tqdm(
        load_embeddings_batch(store, non_error_index, batch_size),
        desc="Computing PCA",
        total=len(non_error_index) // batch_size + 1,
    ):
        ipca.partial_fit(batch)

    # Get explained variance ratio and eigenvectors
    explained_variance_ratio = ipca.explained_variance_ratio_
    eigenvectors = ipca.components_  # Shape will be (300, 300)

    return eigenvectors, explained_variance_ratio, ipca


def direct_egv_comp(store, non_error_index, batch_size=1000):
    """
    Compute eigenvectors through direct covariance matrix calculation
    """
    # Step 1: Calculate mean vector
    print("Computing mean vector...")
    mean_vector = np.zeros(300)
    total_count = 0

    for batch in tqdm(
        load_embeddings_batch(store, non_error_index, batch_size),
        desc="Computing mean",
        total=len(non_error_index) // batch_size + 1,
    ):
        mean_vector += np.sum(batch, axis=0)
        total_count += batch.shape[0]

    mean_vector /= total_count

    # Step 2: Compute covariance matrix block by block
    print("Computing covariance matrix...")
    cov_matrix = np.zeros((300, 300))

    for batch in tqdm(
        load_embeddings_batch(store, non_error_index, batch_size),
        desc="Computing covariance",
        total=len(non_error_index) // batch_size + 1,
    ):
        # Center the batch
        centered_batch = batch - mean_vector
        # Update covariance matrix
        cov_matrix += centered_batch.T @ centered_batch

    cov_matrix /= total_count - 1

    # Step 3: Compute eigenvectors and eigenvalues
    print("Computing eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    return eigenvectors.T, explained_variance_ratio, None  # None instead of ipca
