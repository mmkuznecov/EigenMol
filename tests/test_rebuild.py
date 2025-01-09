import numpy as np
import os
import shutil
from typing import Iterator, Dict, Any
from datetime import datetime
from src.datasets.tensor_storage import TensorStorage


def create_test_data(
    n_samples: int = 300, embedding_size: int = 300
) -> tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """Create test data for TensorStorage with random embeddings."""

    def tensor_iterator() -> Iterator[np.ndarray]:
        for _ in range(n_samples):
            # Create random embedding vector
            yield np.random.randn(embedding_size).astype(np.float32)

    def metadata_iterator() -> Iterator[Dict[str, Any]]:
        for i in range(n_samples):
            yield {
                "vector_id": i,
                "created_at": datetime.now(),
                "batch_id": i // 50,  # Group into batches of 50
                "norm": np.random.uniform(0.8, 1.2),  # Random norm value
                "is_normalized": False,
            }

    return tensor_iterator(), metadata_iterator()


def test_storage_rebuild():
    """Test storage creation and rebuilding functionality."""

    # Parameters
    storage_dir = "test_embeddings_storage"
    n_samples = 100000
    embedding_size = 300

    # Initial chunk size: relatively small for demonstration
    initial_chunk_size = embedding_size * (2**13)  # About 2.4MB

    # Clean up if exists
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    print("\n=== Creating Initial Storage ===")
    print(f"Number of vectors: {n_samples}")
    print(f"Vector size: {embedding_size}")
    print(f"Initial chunk size: {initial_chunk_size / (1024*1024):.2f} MB")

    # Create test data
    tensor_iter, metadata_iter = create_test_data(n_samples, embedding_size)

    # Create storage
    storage = TensorStorage.create_storage(
        storage_dir=storage_dir,
        data_iterator=tensor_iter,
        metadata_iterator=metadata_iter,
        chunk_size=initial_chunk_size,
        description="Test embeddings storage with small chunks",
    )

    print("\nInitial Storage Info:")
    print(storage)

    # Test initial storage
    print("\nTesting initial storage...")
    test_idx = 42  # Random test index
    test_tensor = storage[test_idx]
    test_meta = storage.get_params_for_tensor(test_idx)
    print(f"\nTest vector {test_idx}:")
    print(f"Shape: {test_tensor.shape}")
    print(f"Metadata: {test_meta}")

    # New chunk size: much larger
    new_chunk_size = embedding_size * (2**17)  # About 39MB
    print("\n=== Rebuilding Storage In-Place ===")
    print(f"New chunk size: {new_chunk_size / (1024*1024):.2f} MB")

    # Rebuild in place
    storage.rebuild_storage(
        new_chunk_size=new_chunk_size,
        description="Test embeddings storage with large chunks",
        inplace=True,
    )

    print("\nRebuilt Storage Info:")
    print(storage)

    # Verify data after rebuild
    print("\nVerifying rebuilt storage...")
    rebuilt_tensor = storage[test_idx]
    rebuilt_meta = storage.get_params_for_tensor(test_idx)

    assert np.allclose(test_tensor, rebuilt_tensor), "Data mismatch after rebuild!"
    assert test_meta == rebuilt_meta, "Metadata mismatch after rebuild!"

    print("\nVerification successful!")
    print(f"Test vector {test_idx} after rebuild:")
    print(f"Shape: {rebuilt_tensor.shape}")
    print(f"Metadata: {rebuilt_meta}")

    # Additional storage info
    storage_info = storage.get_storage_info()
    print("\nDetailed Storage Information:")
    for key, value in storage_info.items():
        print(f"{key}: {value}")

    # Clean up
    shutil.rmtree(storage_dir)
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_storage_rebuild()
