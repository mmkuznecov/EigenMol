import click
import logging
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from src.models.mol2vec.mol2vec_encoder import Mol2VecEncoder
from src.datasets.tensor_storage import TensorStorage
from rdkit import RDLogger
import warnings

# Suppress warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")


def process_peptide_embeddings(
    parquet_path: str,
    model_path: str,
    storage_dir: str,
    peptide_type: str,
    chunk_size: Optional[int] = None,
) -> TensorStorage:
    """Process embeddings for a peptide dataset."""
    try:
        # Load data
        df = pd.read_parquet(parquet_path)
        logging.info(f"Loaded {len(df)} {peptide_type} SMILES")

        # Initialize encoder
        encoder = Mol2VecEncoder(model_path)
        os.makedirs(storage_dir, exist_ok=True)

        # Set default chunk size if not provided
        chunk_size = chunk_size or 300 * (2**13)

        def tensor_iterator():
            for smiles in df["SMILES"]:
                embedding = encoder.smiles_to_vec(smiles)
                yield (
                    embedding.astype(np.float32)
                    if embedding is not None
                    else np.zeros(300, dtype=np.float32)
                )

        def metadata_iterator():
            for idx, row in enumerate(df.itertuples()):
                metadata = {
                    "smiles": row.SMILES,
                    "success": encoder.smiles_to_vec(row.SMILES) is not None,
                    "index": idx,
                }
                if hasattr(row, "subdir"):  # For tetrapeptides
                    metadata["subdir"] = row.subdir
                yield metadata

        # Create storage
        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iterator(),
            metadata_iterator=metadata_iterator(),
            chunk_size=chunk_size,
            description=f"Mol2Vec {peptide_type} embeddings",
        )

        # Print statistics
        success_count = storage.metadata_df["success"].sum()
        total_count = len(storage.metadata_df)
        logging.info(f"\n{peptide_type} Statistics:")
        logging.info(f"Total molecules: {total_count}")
        logging.info(f"Successful embeddings: {success_count}")
        logging.info(f"Success rate: {success_count/total_count:.2%}")

        if "subdir" in storage.metadata_df.columns:
            subdirs = storage.metadata_df["subdir"].unique()
            logging.info(f"Unique subdirs: {len(subdirs)}")

        return storage

    except Exception as e:
        logging.error(f"Error processing {peptide_type}: {str(e)}")
        raise


def verify_storage(storage: TensorStorage, peptide_type: str):
    """Verify storage content and integrity."""
    try:
        success_indices = storage.filter_tensors(success=True)
        if success_indices:
            sample = storage[success_indices[0]]
            logging.info(f"\n{peptide_type} Verification:")
            logging.info(f"Sample tensor shape: {sample.shape}")
            logging.info(f"Success rate: {storage.metadata_df['success'].mean():.2%}")

    except Exception as e:
        logging.error(f"Error verifying {peptide_type}: {str(e)}")
        raise


@click.command()
@click.option(
    "--input-dir", required=True, type=str, help="Base directory for peptide data"
)
@click.option("--model-path", required=True, type=str, help="Path to mol2vec model")
@click.option(
    "--output-dir", required=True, type=str, help="Base directory for output storage"
)
@click.option(
    "--peptide-type",
    type=click.Choice(["di", "tri", "tetra", "all"]),
    required=True,
    help="Type of peptides to process",
)
@click.option("--chunk-size", type=int, help="Size of storage chunks (optional)")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    input_dir: str,
    model_path: str,
    output_dir: str,
    peptide_type: str,
    chunk_size: Optional[int],
    log_level: str,
):
    """Generate Mol2Vec embeddings for peptide datasets."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    base_dir = Path(input_dir)
    storage_base = Path(output_dir)

    # Define peptide configurations
    peptide_configs = {
        "di": ("dipeptides", "dipeptides.parquet", "mol2vec_dipeptides"),
        "tri": ("tripeptides", "tripeptides.parquet", "mol2vec_tripeptides"),
        "tetra": ("tetrapeptides", "tetrapeptides.parquet", "mol2vec_tetrapeptides"),
    }

    try:
        # Process selected peptide type(s)
        if peptide_type == "all":
            process_types = list(peptide_configs.keys())
        else:
            process_types = [peptide_type]

        for ptype in process_types:
            config = peptide_configs[ptype]
            peptide_name, parquet_file, storage_name = config

            logging.info(f"\nProcessing {peptide_name}...")
            parquet_path = base_dir / peptide_name / parquet_file
            storage_dir = storage_base / storage_name

            # Process embeddings
            storage = process_peptide_embeddings(
                parquet_path=str(parquet_path),
                model_path=model_path,
                storage_dir=str(storage_dir),
                peptide_type=peptide_name,
                chunk_size=chunk_size,
            )

            # Verify storage
            verify_storage(storage, peptide_name)
            storage.close()

        logging.info("\nAll specified peptides processed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()

# python scripts/m2v_emb_generation_peptides.py \
#     --input-dir data/peptides \
#     --model-path src/models/mol2vec/model_300dim.pkl \
#     --output-dir storages \
#     --peptide-type all \
#     --log-level INFO
