import click
import numpy as np
import pandas as pd
import os
import logging
import warnings
from typing import Iterator, Dict, Any, Optional, Tuple
from tqdm import tqdm
from src.models.mol2vec.mol2vec_encoder import Mol2VecEncoder
from src.datasets.tensor_storage import TensorStorage
from rdkit import RDLogger

# Suppress RDKit and mol2vec warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def create_embedding_iterator(
    df: pd.DataFrame, encoder: Mol2VecEncoder
) -> Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """Create iterators for embeddings and metadata."""

    def tensor_iterator() -> Iterator[np.ndarray]:
        for smiles in tqdm(df["SMILES"], desc="Generating embeddings"):
            embedding = encoder.smiles_to_vec(smiles)
            if embedding is not None:
                yield embedding.astype(np.float32)
            else:
                yield np.zeros(300, dtype=np.float32)

    def metadata_iterator() -> Iterator[Dict[str, Any]]:
        for idx, smiles in enumerate(df["SMILES"]):
            embedding = encoder.smiles_to_vec(smiles)
            yield {"smiles": smiles, "success": embedding is not None, "index": idx}

    return tensor_iterator(), metadata_iterator()


def verify_storage(storage: TensorStorage):
    """Verify storage content and integrity."""
    logging.info("\nVerifying storage...")
    success_indices = storage.filter_tensors(success=True)
    if success_indices:
        sample_idx = success_indices[0]
        sample_tensor = storage[sample_idx]
        logging.info(f"Sample successful embedding shape: {sample_tensor.shape}")

    success_count = storage.metadata_df["success"].sum()
    total_count = len(storage.metadata_df)
    logging.info("\nStorage Statistics:")
    logging.info(f"Total molecules: {total_count}")
    logging.info(f"Successful embeddings: {success_count}")
    logging.info(f"Success rate: {success_count/total_count:.2%}")


@click.command()
@click.option(
    "--input-file",
    required=True,
    type=str,
    help="Path to input parquet file with SMILES",
)
@click.option(
    "--model-path", required=True, type=str, help="Path to mol2vec model file"
)
@click.option(
    "--output-dir", required=True, type=str, help="Directory for output storage"
)
@click.option("--chunk-size", type=int, help="Size of storage chunks (optional)")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    input_file: str,
    model_path: str,
    output_dir: str,
    chunk_size: Optional[int],
    log_level: str,
):
    """Generate Mol2Vec embeddings for DGSM dataset."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Load data
        logging.info(f"Reading parquet file: {input_file}")
        df = pd.read_parquet(input_file)
        logging.info(f"Loaded {len(df)} SMILES strings")

        # Initialize encoder
        logging.info("Initializing Mol2Vec encoder...")
        encoder = Mol2VecEncoder(model_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set default chunk size if not provided
        if chunk_size is None:
            chunk_size = 300 * (2**13)  # Default for 300-dimensional embeddings

        # Create iterators
        tensor_iter, metadata_iter = create_embedding_iterator(df, encoder)

        # Create storage
        logging.info("Creating tensor storage...")
        storage = TensorStorage.create_storage(
            storage_dir=output_dir,
            data_iterator=tensor_iter,
            metadata_iterator=metadata_iter,
            chunk_size=chunk_size,
            description="Mol2Vec DGSM embeddings",
        )

        # Verify storage
        verify_storage(storage)
        storage.close()

        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/m2v_emb_generation_dgsm.py \
#     --input-file data/dgsm/chembl_22_clean_1576904_sorted_std_final.parquet \
#     --model-path src/models/mol2vec/model_300dim.pkl \
#     --output-dir storages/mol2vec_dgsm \
#     --log-level INFO
