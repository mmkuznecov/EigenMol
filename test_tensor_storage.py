import numpy as np
import os
import shutil
import pytest
from typing import Iterator, Dict, Any
import pandas as pd
from datasets.tensor_storage import TensorStorage

def create_test_data(n_samples: int = 10000, embedding_size: int = 300) -> tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """Create test data for TensorStorage."""
    
    def tensor_iterator() -> Iterator[np.ndarray]:
        for _ in range(n_samples):
            yield np.random.randn(embedding_size).astype(np.float32)
    
    def metadata_iterator() -> Iterator[Dict[str, Any]]:
        for i in range(n_samples):
            yield {
                'has_error': bool(np.random.binomial(1, 0.3)),  # 30% chance of error
                'created_at': pd.Timestamp.now(),
                'batch_id': i // 100,  # group into batches of 100
                'confidence_score': np.random.uniform(0.5, 1.0)
            }
    
    return tensor_iterator(), metadata_iterator()

def test_tensor_storage():
    """Test the TensorStorage functionality."""
    # Setup
    storage_dir = "test_storage"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    
    # Create test data
    n_samples = 10000
    embedding_size = 300
    print(f"\nCreating test data with {n_samples} samples of shape ({embedding_size},)")
    
    tensor_iter, metadata_iter = create_test_data(n_samples, embedding_size)
    
    # Calculate chunk size: 300 * 2^18 bytes
    chunk_size = embedding_size * (2 ** 13)
    
    # Create storage
    storage = TensorStorage.create_storage(
        storage_dir=storage_dir,
        data_iterator=tensor_iter,
        metadata_iterator=metadata_iter,
        chunk_size=chunk_size,
        description="Test storage for embeddings"
    )
    
    # Print storage information
    print("\nStorage Information:")
    print(storage)
    
    # Test basic functionality
    assert len(storage) == n_samples, f"Storage should contain {n_samples} elements"
    
    # Test tensor retrieval
    tensor_0 = storage[0]
    print(f"\nSample tensor shape: {tensor_0.shape}")
    assert tensor_0.shape == (embedding_size,), f"Tensor shape should be ({embedding_size},)"
    
    # Test metadata queries
    error_tensors = storage.metadata_df[storage.metadata_df['has_error'] == True]
    print(f"\nNumber of tensors with errors: {len(error_tensors)}")
    assert len(error_tensors) > 0, "Should have some tensors with errors"
    
    # Test tensor retrieval by parameter
    error_tensor = storage.get_tensor_by_param('has_error', True)
    assert error_tensor is not None, "Should be able to retrieve tensor by parameter"
    assert error_tensor.shape == (embedding_size,), f"Retrieved tensor should have shape ({embedding_size},)"
    
    # Get and print storage info
    storage_info = storage.get_storage_info()
    print("\nDetailed Storage Information:")
    for key, value in storage_info.items():
        print(f"{key}: {value}")
    
    # Cleanup
    shutil.rmtree(storage_dir)
    print("\nTests completed successfully!")

if __name__ == "__main__":
    test_tensor_storage()