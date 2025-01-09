import click
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


class GPUMonitor:
    """Monitor GPU memory and utilization"""

    @staticmethod
    def log_gpu_stats():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                cached = torch.cuda.memory_reserved(i) / 1024**2
                logging.info(
                    f"GPU {i} Memory - Allocated: {allocated:.2f}MB, Cached: {cached:.2f}MB"
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
        """Process a batch and return embeddings with success flags."""
        start_time = time.time()
        GPUMonitor.log_gpu_stats()

        inputs = self.encoder.tokenizer(
            smiles_batch, padding=True, return_tensors="pt", truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder.model(**inputs)

        embeddings = outputs["pooler_output"].cpu().numpy()
        success_flags = [True] * len(smiles_batch)

        if self.device == "cuda":
            torch.cuda.synchronize()

        logging.debug(
            f"Batch processing time: {time.time() - start_time:.2f}s, "
            f"Batch size: {len(smiles_batch)}"
        )

        return embeddings, success_flags


def process_peptide_embeddings(
    parquet_path: str,
    storage_dir: str,
    peptide_type: str,
    batch_processor: BatchProcessor,
    batch_size: int = 32,
    chunk_size: Optional[int] = None,
) -> TensorStorage:
    """Process embeddings for a peptide dataset."""
    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Loaded {len(df)} {peptide_type} SMILES")

        os.makedirs(storage_dir, exist_ok=True)
        chunk_size = chunk_size or 768 * (2**13)

        total_batches = (len(df) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=total_batches, desc=f"Processing {peptide_type}")
        processed_batches = {}

        def process_and_cache_batch(batch_idx: int) -> Tuple[np.ndarray, List[bool]]:
            if batch_idx not in processed_batches:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(df))
                batch_smiles = df["SMILES"].iloc[start_idx:end_idx].tolist()

                embeddings, success_flags = batch_processor.process_batch(batch_smiles)
                processed_batches[batch_idx] = (embeddings, success_flags)
                progress_bar.update(1)

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
                        "processing_time": time.time(),
                    }
                    if "subdir" in df.columns:
                        metadata["subdir"] = df.iloc[start_idx + j]["subdir"]
                    yield metadata

        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iterator(),
            metadata_iterator=metadata_iterator(),
            chunk_size=chunk_size,
            description=f"MolFormer {peptide_type} embeddings",
        )

        progress_bar.close()
        return storage

    except Exception as e:
        logging.error(f"Error processing {peptide_type}: {str(e)}")
        raise


def verify_storage(storage: TensorStorage, peptide_type: str):
    """Verify storage content and integrity."""
    logging.info(f"\nVerifying {peptide_type} storage...")

    success_indices = storage.filter_tensors(success=True)
    if success_indices:
        sample_tensor = storage[success_indices[0]]
        logging.info(f"Sample tensor shape: {sample_tensor.shape}")

    batch_stats = storage.metadata_df.groupby("batch_id").agg(
        {"success": ["count", "sum", "mean"], "processing_time": ["min", "max", "mean"]}
    )

    logging.info("\nBatch Statistics:")
    logging.info(f"Total batches: {len(batch_stats)}")
    logging.info(
        f"Average batch success rate: {batch_stats['success']['mean'].mean():.2%}"
    )

    success_count = storage.metadata_df["success"].sum()
    total_count = len(storage.metadata_df)
    logging.info(f"\n{peptide_type} Statistics:")
    logging.info(f"Total molecules: {total_count}")
    logging.info(f"Successful embeddings: {success_count}")
    logging.info(f"Success rate: {success_count/total_count:.2%}")

    if "subdir" in storage.metadata_df.columns:
        subdirs = storage.metadata_df["subdir"].unique()
        logging.info(f"Unique subdirs: {len(subdirs)}")


@click.command()
@click.option(
    "--input-dir", required=True, type=str, help="Base directory for peptide data"
)
@click.option(
    "--output-dir", required=True, type=str, help="Base directory for output storage"
)
@click.option(
    "--peptide-type",
    type=click.Choice(["di", "tri", "tetra", "all"]),
    required=True,
    help="Type of peptides to process",
)
@click.option("--batch-size", default=256, type=int, help="Batch size for processing")
@click.option("--chunk-size", type=int, help="Size of storage chunks (optional)")
@click.option("--device", default="cuda", type=str, help="Device to use (cuda/cpu)")
@click.option("--log-level", default="INFO", type=str, help="Logging level")
def main(
    input_dir: str,
    output_dir: str,
    peptide_type: str,
    batch_size: int,
    chunk_size: Optional[int],
    device: str,
    log_level: str,
):
    """Generate MolFormer embeddings for peptide datasets."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Print system information
    logging.info("\nSystem Configuration:")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"Number of CPUs: {psutil.cpu_count()}")
    logging.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    base_dir = Path(input_dir)
    storage_base = Path(output_dir)

    # Define peptide configurations
    peptide_configs = {
        "di": ("dipeptides", "dipeptides.parquet", "molformer_dipeptides"),
        "tri": ("tripeptides", "tripeptides.parquet", "molformer_tripeptides"),
        "tetra": ("tetrapeptides", "tetrapeptides.parquet", "molformer_tetrapeptides"),
    }

    try:
        # Initialize encoder and processor
        encoder = MolformerEncoder()
        batch_processor = BatchProcessor(encoder, device)

        # Process selected peptide type(s)
        if peptide_type == "all":
            process_types = list(peptide_configs.keys())
        else:
            process_types = [peptide_type]

        for ptype in process_types:
            start_time = time.time()
            config = peptide_configs[ptype]
            peptide_name, parquet_file, storage_name = config

            logging.info(f"\nProcessing {peptide_name}...")
            parquet_path = base_dir / peptide_name / parquet_file
            storage_dir = storage_base / storage_name

            # Process embeddings
            storage = process_peptide_embeddings(
                parquet_path=str(parquet_path),
                storage_dir=str(storage_dir),
                peptide_type=peptide_name,
                batch_processor=batch_processor,
                batch_size=batch_size,
                chunk_size=chunk_size,
            )

            # Verify storage
            verify_storage(storage, peptide_name)
            storage.close()

            end_time = time.time()
            logging.info(
                f"{peptide_name} processing time: {end_time - start_time:.2f}s"
            )

            # Clear GPU cache between peptide types
            GPUMonitor.clear_gpu_cache()

        logging.info("\nAll specified peptides processed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# python scripts/molformer_emb_generation_peptides.py \
#     --input-dir data/peptides \
#     --output-dir storages \
#     --peptide-type all \
#     --batch-size 256 \
#     --device cuda \
#     --log-level INFO
