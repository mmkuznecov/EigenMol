import numpy as np
import pandas as pd
import os
import logging
import warnings
import torch
from typing import Iterator, Dict, Any, Optional, Tuple
from tqdm import tqdm
from models.molformer.molformer_encoder import MolformerEncoder
from datasets.tensor_storage import TensorStorage

# Configure logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


def process_smiles_batch(
    encoder: MolformerEncoder, smiles_batch: list, device: str = "cuda"
) -> Tuple[np.ndarray, list]:
    """
    Process a batch of SMILES strings, falling back to individual processing on errors

    Args:
        encoder (MolformerEncoder): MolFormer encoder
        smiles_batch (list): Batch of SMILES strings
        device (str): Device to use for processing

    Returns:
        Tuple[np.ndarray, list]: Embeddings and success flags
    """
    try:
        # Try batch processing first
        batch_embeddings = encoder.batch_embed(
            smiles_batch, batch_size=len(smiles_batch)
        )
        if batch_embeddings is not None:
            return batch_embeddings, [True] * len(smiles_batch)
    except Exception as e:
        logging.debug(
            f"Batch processing failed: {str(e)}, falling back to individual processing"
        )

    # Fall back to individual processing
    embeddings = []
    success_flags = []

    for smiles in smiles_batch:
        try:
            embedding = encoder.get_embedding(smiles)
            if embedding is not None:
                embeddings.append(embedding)
                success_flags.append(True)
            else:
                embeddings.append(np.zeros(768, dtype=np.float32))
                success_flags.append(False)
        except Exception as e:
            embeddings.append(np.zeros(768, dtype=np.float32))
            success_flags.append(False)
            logging.debug(f"Error processing SMILES {smiles}: {str(e)}")

    return np.array(embeddings), success_flags


def create_embedding_iterator(
    df: pd.DataFrame, encoder: MolformerEncoder, batch_size: int = 32
) -> Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """
    Create iterators for embeddings and metadata

    Args:
        df (pd.DataFrame): DataFrame with SMILES strings
        encoder (MolformerEncoder): Initialized MolFormer encoder
        batch_size (int): Size of batches for processing

    Returns:
        Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]: Iterators for data and metadata
    """
    total_batches = (len(df) + batch_size - 1) // batch_size

    def tensor_iterator() -> Iterator[np.ndarray]:
        for i in tqdm(
            range(0, len(df), batch_size),
            total=total_batches,
            desc="Processing batches",
        ):
            batch_smiles = df["SMILES"].iloc[i : i + batch_size].tolist()
            embeddings, _ = process_smiles_batch(encoder, batch_smiles)
            for emb in embeddings:
                yield emb.astype(np.float32)

    def metadata_iterator() -> Iterator[Dict[str, Any]]:
        for i in range(0, len(df), batch_size):
            batch_smiles = df["SMILES"].iloc[i : i + batch_size].tolist()
            _, success_flags = process_smiles_batch(encoder, batch_smiles)

            for j, (smiles, success) in enumerate(zip(batch_smiles, success_flags)):
                yield {
                    "smiles": smiles,
                    "success": success,
                    "index": i + j,
                    "batch_id": i // batch_size,
                }

    return tensor_iterator(), metadata_iterator()


def process_and_store_embeddings(
    parquet_path: str,
    storage_dir: str,
    batch_size: int = 256,
    chunk_size: Optional[int] = None,
) -> TensorStorage:
    """
    Process SMILES strings and store embeddings in TensorStorage

    Args:
        parquet_path (str): Path to input parquet file with SMILES
        storage_dir (str): Directory for tensor storage
        batch_size (int): Batch size for processing
        chunk_size (Optional[int]): Size of storage chunks
    """
    try:
        # Load data
        logging.info(f"Reading parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logging.info(f"Loaded {len(df)} SMILES strings")

        # Initialize encoder
        logging.info("Initializing MolFormer encoder...")
        encoder = MolformerEncoder()

        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)

        # Calculate default chunk size if not provided
        if chunk_size is None:
            chunk_size = 768 * (2**13)  # Suitable for 768-dimensional embeddings

        # Create iterators
        tensor_iter, metadata_iter = create_embedding_iterator(df, encoder, batch_size)

        # Create storage
        logging.info("Creating tensor storage...")
        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iter,
            metadata_iterator=metadata_iter,
            chunk_size=chunk_size,
            description="MolFormer embeddings storage",
        )

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

        # Print batch statistics
        batch_counts = storage.metadata_df.groupby("batch_id")["success"].agg(
            ["count", "sum"]
        )
        logging.info("\nBatch Statistics:")
        logging.info(f"Total batches: {len(batch_counts)}")
        logging.info(
            f"Average success rate per batch: {batch_counts['sum'].mean() / batch_counts['count'].mean():.2%}"
        )

    except Exception as e:
        logging.error(f"Error in verification: {str(e)}")
        raise


def main():
    """Main execution function"""
    # Configuration
    PARQUET_FILE = "data/dgsm/chembl_22_clean_1576904_sorted_std_final.parquet"
    STORAGE_DIR = "storages/molformer_dgsm"
    BATCH_SIZE = 32  # Adjust based on your GPU memory

    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Process and store embeddings
        storage = process_and_store_embeddings(
            parquet_path=PARQUET_FILE, storage_dir=STORAGE_DIR, batch_size=BATCH_SIZE
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
