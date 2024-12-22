import logging
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from src.models.mol2vec.mol2vec_encoder import Mol2VecEncoder
from src.datasets.tensor_storage import TensorStorage
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_peptide_embeddings(
    parquet_path: str,
    model_path: str,
    storage_dir: str,
    peptide_type: str,
    chunk_size: Optional[int] = None,
) -> TensorStorage:
    """Process embeddings for a peptide dataset"""
    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Loaded {len(df)} {peptide_type} SMILES")

        encoder = Mol2VecEncoder(model_path)
        os.makedirs(storage_dir, exist_ok=True)

        # Default chunk size for 300-dim embeddings
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
            for idx, row in df.iterrows():
                metadata = {
                    "smiles": row["SMILES"],
                    "success": encoder.smiles_to_vec(row["SMILES"]) is not None,
                    "index": idx,
                }
                if "subdir" in row:  # For tetrapeptides
                    metadata["subdir"] = row["subdir"]
                yield metadata

        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iterator(),
            metadata_iterator=metadata_iterator(),
            chunk_size=chunk_size,
            description=f"Mol2Vec {peptide_type} embeddings",
        )

        # Print statistics
        success_count = storage.metadata_df["success"].sum()
        logging.info(f"\n{peptide_type} Statistics:")
        logging.info(f"Total: {len(storage.metadata_df)}")
        logging.info(f"Success: {success_count}")
        logging.info(f"Failed: {len(storage.metadata_df) - success_count}")

        if "subdir" in storage.metadata_df:
            subdirs = storage.metadata_df["subdir"].unique()
            logging.info(f"Unique subdirs: {len(subdirs)}")

        return storage

    except Exception as e:
        logging.error(f"Error processing {peptide_type}: {e}")
        raise


def verify_storage(storage: TensorStorage, peptide_type: str):
    """Verify storage content"""
    try:
        success_indices = storage.filter_tensors(success=True)
        if success_indices:
            sample = storage[success_indices[0]]
            logging.info(f"\n{peptide_type} Verification:")
            logging.info(f"Sample shape: {sample.shape}")
            logging.info(f"Success rate: {storage.metadata_df['success'].mean():.2%}")

    except Exception as e:
        logging.error(f"Error verifying {peptide_type}: {e}")
        raise


def main():
    BASE_DIR = Path("data/peptides")
    MODEL_PATH = "models/mol2vec/model_300dim.pkl"
    STORAGE_BASE = Path("storages")

    peptide_configs = [
        ("dipeptides", "dipeptides.parquet", "mol2vec_dipeptides"),
        ("tripeptides", "tripeptides.parquet", "mol2vec_tripeptides"),
        ("tetrapeptides", "tetrapeptides.parquet", "mol2vec_tetrapeptides"),
    ]

    try:
        for peptide_type, parquet_file, storage_name in peptide_configs:
            parquet_path = BASE_DIR / peptide_type / parquet_file
            storage_dir = STORAGE_BASE / storage_name

            logging.info(f"\nProcessing {peptide_type}...")

            storage = process_peptide_embeddings(
                parquet_path=str(parquet_path),
                model_path=MODEL_PATH,
                storage_dir=str(storage_dir),
                peptide_type=peptide_type,
            )

            verify_storage(storage, peptide_type)
            storage.close()

        logging.info("\nAll peptides processed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
