import numpy as np
import pandas as pd
import os
import logging
import warnings
from typing import Iterator, Dict, Any, Optional, Tuple
from tqdm import tqdm
from models.mol2vec.mol2vec_encoder import Mol2VecEncoder
from datasets.tensor_storage import TensorStorage
from rdkit import RDLogger

# Suppress all RDKit logging
RDLogger.DisableLog("rdApp.*")

# Specifically suppress the MorganGenerator deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")

# Configure logging to only show INFO and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


def create_embedding_iterator(
    df: pd.DataFrame, encoder: Mol2VecEncoder
) -> Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """
    Create iterators for embeddings and metadata

    Args:
        df (pd.DataFrame): DataFrame with SMILES strings
        encoder (Mol2VecEncoder): Initialized Mol2Vec encoder

    Returns:
        Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]: Iterators for data and metadata
    """

    def tensor_iterator() -> Iterator[np.ndarray]:
        for smiles in tqdm(df["SMILES"], desc="Generating embeddings"):
            embedding = encoder.smiles_to_vec(smiles)
            if embedding is not None:
                yield embedding.astype(np.float32)
            else:
                # Return zero vector for failed embeddings
                yield np.zeros(300, dtype=np.float32)

    def metadata_iterator() -> Iterator[Dict[str, Any]]:
        for idx, smiles in enumerate(df["SMILES"]):
            embedding = encoder.smiles_to_vec(smiles)
            yield {"smiles": smiles, "success": embedding is not None, "index": idx}

    return tensor_iterator(), metadata_iterator()


def process_and_store_embeddings(
    parquet_path: str,
    model_path: str,
    storage_dir: str,
    chunk_size: Optional[int] = None,
) -> TensorStorage:
    """
    Process SMILES strings and store embeddings in TensorStorage

    Args:
        parquet_path (str): Path to input parquet file with SMILES
        model_path (str): Path to mol2vec model
        storage_dir (str): Directory for tensor storage
        chunk_size (Optional[int]): Size of storage chunks

    Returns:
        TensorStorage: Created storage instance
    """
    try:
        # Load data
        logging.info(f"Reading parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logging.info(f"Loaded {len(df)} SMILES strings")

        # Initialize encoder
        logging.info("Initializing Mol2Vec encoder...")
        encoder = Mol2VecEncoder(model_path)

        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)

        # Calculate default chunk size if not provided
        if chunk_size is None:
            chunk_size = 300 * (2**13)  # Suitable for 300-dimensional embeddings

        # Create iterators
        tensor_iter, metadata_iter = create_embedding_iterator(df, encoder)

        # Create storage
        logging.info("Creating tensor storage...")
        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iter,
            metadata_iterator=metadata_iter,
            chunk_size=chunk_size,
            description="Mol2Vec embeddings storage",
        )

        # Verify storage
        logging.info("\nStorage Information:")
        print(storage)

        # Print statistics
        success_count = storage.metadata_df["success"].sum()
        total_count = len(storage.metadata_df)
        logging.info(f"\nProcessing Statistics:")
        logging.info(f"Total molecules: {total_count}")
        logging.info(f"Successful embeddings: {success_count}")
        logging.info(f"Failed embeddings: {total_count - success_count}")

        return storage

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        raise


def verify_storage(storage: TensorStorage):
    """
    Verify the created storage

    Args:
        storage (TensorStorage): Storage to verify
    """
    try:
        logging.info("\nVerifying storage...")

        # Check successful embeddings
        success_indices = storage.filter_tensors(success=True)
        if success_indices:
            sample_idx = success_indices[0]
            sample_tensor = storage[sample_idx]
            logging.info(f"Sample successful embedding shape: {sample_tensor.shape}")

        # Check failed embeddings
        failed_indices = storage.filter_tensors(success=False)
        if failed_indices:
            sample_idx = failed_indices[0]
            sample_tensor = storage[sample_idx]
            logging.info(f"Sample failed embedding shape: {sample_tensor.shape}")

        # Print some metadata statistics
        logging.info("\nMetadata Statistics:")
        logging.info(f"Total entries: {len(storage.metadata_df)}")
        logging.info(f"Success rate: {storage.metadata_df['success'].mean():.2%}")

    except Exception as e:
        logging.error(f"Error in verification: {str(e)}")
        raise


def main():
    """Main execution function"""
    # Configuration
    PARQUET_FILE = "data/dgsm/chembl_22_clean_1576904_sorted_std_final.parquet"
    MODEL_PATH = "models/mol2vec/model_300dim.pkl"  # Update with your model path
    STORAGE_DIR = "storages/mol2vec_dgsm"

    try:
        # Process and store embeddings
        storage = process_and_store_embeddings(
            parquet_path=PARQUET_FILE, model_path=MODEL_PATH, storage_dir=STORAGE_DIR
        )

        # Verify storage
        verify_storage(storage)

        # Close storage
        storage.close()

        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
