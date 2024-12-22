import numpy as np
from typing import Union, Tuple, Iterator, Optional
import warnings
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import torch


class BatchIterator:
    def __init__(self, store, indices, batch_size):
        """
        Iterator to load embeddings in batches.

        Args:
            store: Data storage containing embeddings
            indices: Indices to iterate over
            batch_size: Size of batches to yield
        """
        self.store = store
        self.indices = indices
        self.batch_size = batch_size
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[
            self.current : min(self.current + self.batch_size, len(self.indices))
        ]
        self.current += self.batch_size

        # Collect batch
        batch = []
        for idx in batch_indices:
            emb = self.store[idx]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            batch.append(emb.reshape(1, -1))  # Ensure 2D shape

        return np.vstack(batch)

    def reset(self):
        """Reset iterator to beginning"""
        self.current = 0


class EmbeddingMetrics:
    def __init__(self, embedding_dim: int):
        """
        Initialize embedding quality metrics calculator.

        Args:
            embedding_dim: Dimensionality of embeddings
        """
        self.embedding_dim = embedding_dim
        self.ipca = None
        self.singular_values = None
        self.singular_vectors = None

    def fit(
        self, batch_iterator: BatchIterator, n_samples: int, use_tqdm: bool = True
    ) -> None:
        """
        Compute SVD of the embedding matrix incrementally.

        Args:
            batch_iterator: Iterator yielding embedding batches
            n_samples: Total number of samples
            use_tqdm: Whether to show progress bar
        """
        # Initialize IncrementalPCA
        self.ipca = IncrementalPCA(n_components=self.embedding_dim)

        # Process batches
        iterator = (
            tqdm(batch_iterator, desc="Computing SVD") if use_tqdm else batch_iterator
        )
        for batch in iterator:
            self.ipca.partial_fit(batch)

        # Store singular values and vectors
        self.singular_values = np.sqrt(self.ipca.explained_variance_)
        self.singular_vectors = self.ipca.components_.T

    def rank_me(self) -> float:
        """
        Compute RankMe metric (entropy of normalized singular values).
        """
        if self.singular_values is None:
            raise ValueError("Must call fit() first")

        # Normalize singular values
        p = self.singular_values / np.sum(self.singular_values)

        # Compute entropy
        return -np.sum(p * np.log(p + np.finfo(float).eps))

    def ne_sum(self) -> float:
        """
        Compute NESum using eigenvalues of covariance matrix C = UAUᵀ.
        Following Definition 2.2:
        NESum(M) = Σ λᵢ/λ₀
        where λᵢ are eigenvalues of covariance matrix
        """
        if self.singular_values is None:
            raise ValueError("Must call fit() first")

        # Use eigenvalues directly from PCA
        eigenvalues = self.ipca.explained_variance_

        # Handle zero division
        if eigenvalues[0] == 0:
            return 0

        # Calculate NESum as per Definition 2.2
        return np.sum(eigenvalues / eigenvalues[0])

    def stable_rank(self) -> float:
        """
        Compute stable rank (squared Frobenius norm / squared spectral norm).
        """
        if self.singular_values is None:
            raise ValueError("Must call fit() first")

        frob_norm_squared = np.sum(self.singular_values**2)
        spec_norm_squared = self.singular_values[0] ** 2
        return frob_norm_squared / spec_norm_squared

    def condition_number(self) -> float:
        """
        Compute condition number (ratio of largest to smallest singular value).
        """
        if self.singular_values is None:
            raise ValueError("Must call fit() first")

        # Use only non-zero singular values
        non_zero_vals = self.singular_values[self.singular_values > np.finfo(float).eps]

        if len(non_zero_vals) == 0:
            return np.inf

        return non_zero_vals[0] / non_zero_vals[-1]

    def coherence(self) -> float:
        """
        Compute µ0-incoherence parameter as defined in Definition 3.1.

        For matrix M with SVD M = UΣV⊤, computes:
        µ0 = max(
            (n1/r) * max(1≤i≤n1) ||U⊤ei||²,
            (n2/r) * max(1≤i≤n2) ||V⊤ej||²
        )
        where ei, ej are standard basis vectors.

        Returns:
            float: Coherence parameter µ0
        """
        if self.singular_vectors is None:
            raise ValueError("Must call fit() first")

        # Get effective rank
        r = np.sum(self.singular_values > 1e-10)
        if r == 0:
            return float("inf")

        # Use significant singular vectors
        U = self.singular_vectors[:, :r]

        # Get dimensions
        n1 = U.shape[0]  # Number of samples

        # Compute U coherence
        # Row norms of U are equivalent to ||U⊤ei||²
        u_norms = np.sum(U * U, axis=1)
        u_coherence = (n1 / r) * np.max(u_norms)

        # For embeddings, we typically only need U coherence
        # as V coherence relates to feature space
        return u_coherence

    def compute_all_metrics(self) -> dict:
        """
        Compute all embedding quality metrics.

        Returns:
            dict: Dictionary containing computed metrics
        """
        return {
            "rankme": self.rank_me(),
            "nesum": self.ne_sum(),
            "stable_rank": self.stable_rank(),
            "condition_number": self.condition_number(),
            "coherence": self.coherence(),
        }
