import logging
import os
import torch
import psutil
import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.models.molformer.molformer_encoder import MolformerEncoder
from src.datasets.tensor_storage import TensorStorage
import gc

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    force=True,
)


class GPUMonitor:
    """Monitor GPU memory and utilization"""

    @staticmethod
    def log_gpu_stats():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                cached = torch.cuda.memory_reserved(i) / 1024**2
                logging.info(
                    f"GPU {i} Memory: Allocated: {allocated:.2f}MB, Cached: {cached:.2f}MB"
                )

    @staticmethod
    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class BatchProcessor:
    """Efficient batch processing with monitoring"""

    def __init__(self, encoder: MolformerEncoder, device: str = "cuda"):
        self.encoder = encoder
        self.device = device
        if device == "cuda":
            self.encoder.model.to(device)

    def process_batch(self, smiles_batch: List[str]) -> Tuple[np.ndarray, List[bool]]:
        """Process a single batch with detailed monitoring"""
        start_time = time.time()
        GPUMonitor.log_gpu_stats()

        # Move inputs to GPU
        inputs = self.encoder.tokenizer(
            smiles_batch, padding=True, return_tensors="pt", truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Process batch
        # with torch.cuda.amp.autocast():  # Use automatic mixed precision
        with torch.no_grad():
            outputs = self.encoder.model(**inputs)

        # Get embeddings
        embeddings = outputs["pooler_output"].cpu().numpy()
        success_flags = [True] * len(smiles_batch)

        # Force CUDA synchronization
        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        batch_time = end_time - start_time
        logging.debug(
            f"Batch processing time: {batch_time:.2f}s, Batch size: {len(smiles_batch)}"
        )

        return embeddings, success_flags


def create_efficient_iterators(
    df: pd.DataFrame,
    batch_processor: BatchProcessor,
    batch_size: int,
) -> Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """Create efficient iterators with shared batch processing"""

    total_batches = (len(df) + batch_size - 1) // batch_size
    processed_batches = {}  # Cache for processed batches

    def process_and_cache_batch(batch_idx: int) -> Tuple[np.ndarray, List[bool]]:
        if batch_idx not in processed_batches:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_smiles = df["SMILES"].iloc[start_idx:end_idx].tolist()

            start_time = time.time()
            embeddings, success_flags = batch_processor.process_batch(batch_smiles)
            end_time = time.time()

            logging.info(
                f"Batch {batch_idx + 1}/{total_batches} - "
                f"Time: {end_time - start_time:.2f}s - "
                f"Success: {sum(success_flags)}/{len(success_flags)}"
            )

            processed_batches[batch_idx] = (embeddings, success_flags)

        return processed_batches[batch_idx]

    def tensor_iterator() -> Iterator[np.ndarray]:
        for batch_idx in range(total_batches):
            embeddings, _ = process_and_cache_batch(batch_idx)
            yield from embeddings

    def metadata_iterator() -> Iterator[Dict[str, Any]]:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            batch_smiles = (
                df["SMILES"].iloc[start_idx : start_idx + batch_size].tolist()
            )
            _, success_flags = process_and_cache_batch(batch_idx)

            for j, (smiles, success) in enumerate(zip(batch_smiles, success_flags)):
                metadata = {
                    "smiles": smiles,
                    "success": success,
                    "index": start_idx + j,
                    "batch_id": batch_idx,
                    "processing_time": time.time(),  # Add timestamp
                }

                # Add subdir for tetrapeptides if present
                if "subdir" in df.columns:
                    metadata["subdir"] = df.iloc[start_idx + j]["subdir"]

                yield metadata

    return tensor_iterator(), metadata_iterator()


def process_peptide_embeddings(
    parquet_path: str,
    storage_dir: str,
    peptide_type: str,
    batch_size: int = 32,
    chunk_size: Optional[int] = None,
) -> TensorStorage:
    """Process embeddings with enhanced monitoring"""

    # Load data
    logging.info(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded {len(df)} {peptide_type} SMILES strings")

    # Initialize processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = MolformerEncoder()
    batch_processor = BatchProcessor(encoder, device)

    # Monitor system resources
    logging.info(f"CPU Usage: {psutil.cpu_percent()}%")
    logging.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
    GPUMonitor.log_gpu_stats()

    # Create storage directory
    os.makedirs(storage_dir, exist_ok=True)

    # Set chunk size
    chunk_size = chunk_size or 768 * (2**13)

    # Create iterators
    tensor_iter, metadata_iter = create_efficient_iterators(
        df, batch_processor, batch_size
    )

    # Create storage with progress monitoring
    logging.info("Creating tensor storage...")
    start_time = time.time()

    storage = TensorStorage.create_storage(
        storage_dir=storage_dir,
        data_iterator=tensor_iter,
        metadata_iterator=metadata_iter,
        chunk_size=chunk_size,
        description=f"MolFormer {peptide_type} embeddings",
    )

    end_time = time.time()
    logging.info(f"Storage creation time: {end_time - start_time:.2f}s")

    # Print detailed statistics
    success_count = storage.metadata_df["success"].sum()
    total_count = len(storage.metadata_df)

    logging.info(f"\n{peptide_type} Statistics:")
    logging.info(f"Total molecules: {total_count}")
    logging.info(f"Successful embeddings: {success_count}")
    logging.info(f"Failed embeddings: {total_count - success_count}")
    logging.info(f"Success rate: {success_count/total_count:.2%}")

    if "subdir" in storage.metadata_df.columns:
        subdirs = storage.metadata_df["subdir"].unique()
        logging.info(f"Unique subdirs: {len(subdirs)}")

    GPUMonitor.log_gpu_stats()
    return storage


def verify_storage(storage: TensorStorage, peptide_type: str):
    """Verify storage with detailed checking"""
    logging.info(f"\nVerifying {peptide_type} storage...")

    # Check tensor shapes and types
    success_indices = storage.filter_tensors(success=True)
    if success_indices:
        sample_tensor = storage[success_indices[0]]
        logging.info(f"Sample tensor shape: {sample_tensor.shape}")
        logging.info(f"Sample tensor dtype: {sample_tensor.dtype}")
        logging.info(
            f"Sample tensor range: [{sample_tensor.min():.3f}, {sample_tensor.max():.3f}]"
        )

    # Analyze batch statistics
    batch_stats = storage.metadata_df.groupby("batch_id").agg(
        {"success": ["count", "sum", "mean"], "processing_time": ["min", "max", "mean"]}
    )

    logging.info("\nBatch Statistics:")
    logging.info(f"Total batches: {len(batch_stats)}")
    logging.info(
        f"Average batch success rate: {batch_stats['success']['mean'].mean():.2%}"
    )
    logging.info(
        f"Average batch processing time: {batch_stats['processing_time']['mean'].mean():.2f}s"
    )


def main():
    """Main execution with enhanced monitoring"""
    BASE_DIR = Path("data/peptides")
    STORAGE_BASE = Path("storages")
    BATCH_SIZE = 256

    peptide_configs = [
        ("dipeptides", "dipeptides.parquet", "molformer_dipeptides"),
        ("tripeptides", "tripeptides.parquet", "molformer_tripeptides"),
        ("tetrapeptides", "tetrapeptides.parquet", "molformer_tetrapeptides"),
    ]

    # Initial system status
    logging.info("\nSystem Configuration:")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"Number of CPUs: {psutil.cpu_count()}")
    logging.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    for peptide_type, parquet_file, storage_name in peptide_configs:
        start_time = time.time()
        logging.info(f"\nProcessing {peptide_type}...")

        parquet_path = BASE_DIR / peptide_type / parquet_file
        storage_dir = STORAGE_BASE / storage_name

        storage = process_peptide_embeddings(
            parquet_path=str(parquet_path),
            storage_dir=str(storage_dir),
            peptide_type=peptide_type,
            batch_size=BATCH_SIZE,
        )

        verify_storage(storage, peptide_type)
        storage.close()

        end_time = time.time()
        logging.info(
            f"{peptide_type} total processing time: {end_time - start_time:.2f}s"
        )

        # Clear GPU cache between peptide types
        GPUMonitor.clear_gpu_cache()

    logging.info("\nAll peptides processed successfully!")


if __name__ == "__main__":
    main()
