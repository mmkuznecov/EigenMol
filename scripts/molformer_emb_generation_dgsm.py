import click
import logging
import os
import torch
import psutil
import time
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


def create_efficient_iterators(
    df: pd.DataFrame,
    batch_processor: BatchProcessor,
    batch_size: int,
) -> Tuple[Iterator[np.ndarray], Iterator[Dict[str, Any]]]:
    """Create iterators for tensor data and metadata."""
    total_batches = (len(df) + batch_size - 1) // batch_size
    processed_batches = {}
    progress_bar = tqdm(total=total_batches, desc="Processing batches")

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
                yield {
                    "smiles": smiles,
                    "success": success,
                    "index": start_idx + j,
                    "batch_id": batch_idx,
                    "processing_time": time.time(),
                }

    return tensor_iterator(), metadata_iterator()


def verify_storage(storage: TensorStorage):
    """Verify storage content and integrity."""
    logging.info("\nVerifying storage...")

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


@click.command()
@click.option(
    "--input-file", required=True, type=str, help="Path to input parquet file"
)
@click.option(
    "--output-dir", required=True, type=str, help="Directory for output storage"
)
@click.option("--batch-size", default=256, type=int, help="Batch size for processing")
@click.option("--chunk-size", type=int, help="Size of storage chunks (optional)")
@click.option("--device", default="cuda", type=str, help="Device to use (cuda/cpu)")
@click.option("--log-level", default="INFO", type=str, help="Logging level")
def main(
    input_file: str,
    output_dir: str,
    batch_size: int,
    chunk_size: Optional[int],
    device: str,
    log_level: str,
):
    """Generate MolFormer embeddings for DGSM dataset."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("\nSystem Configuration:")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"Number of CPUs: {psutil.cpu_count()}")
    logging.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    try:
        logging.info(f"Reading parquet file: {input_file}")
        df = pd.read_parquet(input_file)
        logging.info(f"Loaded {len(df)} SMILES strings")

        encoder = MolformerEncoder()
        batch_processor = BatchProcessor(encoder, device)
        os.makedirs(output_dir, exist_ok=True)

        chunk_size = chunk_size or 768 * (2**13)

        tensor_iter, metadata_iter = create_efficient_iterators(
            df, batch_processor, batch_size
        )

        start_time = time.time()
        storage = TensorStorage.create_storage(
            storage_dir=output_dir,
            data_iterator=tensor_iter,
            metadata_iterator=metadata_iter,
            chunk_size=chunk_size,
            description="MolFormer DGSM embeddings",
        )

        verify_storage(storage)
        storage.close()

        GPUMonitor.clear_gpu_cache()

        end_time = time.time()
        logging.info(f"Total processing time: {end_time - start_time:.2f}s")
        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# python scripts/molformer_emb_generation_dgsm.py \
#     --input-file data/dgsm/chembl_22_clean_1576904_sorted_std_final.parquet \
#     --output-dir storages/molformer_dgsm \
#     --batch-size 256 \
#     --device cuda \
#     --log-level INFO
