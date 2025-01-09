import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from datetime import datetime
from emb_quality import EmbeddingMetrics, BatchIterator
from src.datasets.tensor_storage import TensorStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class StorageMetricsAnalyzer:
    """Analyze embedding metrics for different storages"""

    def __init__(self, storage_base_path: str = "storages"):
        self.storage_base = Path(storage_base_path)
        self.storage_configs = {
            "mol2vec": {
                "dim": 300,
                "storages": ["dgsm", "dipeptides", "tripeptides", "tetrapeptides"],
            },
            "molformer": {
                "dim": 768,
                "storages": ["dgsm", "dipeptides", "tripeptides", "tetrapeptides"],
            },
        }

    def analyze_storage(self, storage_path: Path, embedding_dim: int) -> dict:
        """Analyze a single storage and return metrics"""
        logging.info(f"Analyzing storage: {storage_path}")

        try:
            # Load storage
            store = TensorStorage(str(storage_path))
            metadata_df = store.load_metadata_table()

            # Get successful embeddings
            non_error_index = metadata_df[metadata_df["success"] == True].index
            logging.info(f"Found {len(non_error_index)} successful embeddings")

            # Initialize calculator and iterator
            calc = EmbeddingMetrics(embedding_dim=embedding_dim)
            iterator = BatchIterator(store, non_error_index, batch_size=10000)

            # Fit the metrics calculator
            calc.fit(iterator, n_samples=len(non_error_index))

            # Compute metrics
            metrics = calc.compute_all_metrics()

            # Add basic stats
            metrics.update(
                {
                    "total_embeddings": len(metadata_df),
                    "successful_embeddings": len(non_error_index),
                    "success_rate": len(non_error_index) / len(metadata_df),
                }
            )

            store.close()
            return metrics

        except Exception as e:
            logging.error(f"Error analyzing {storage_path}: {str(e)}")
            return None

    def run_analysis(self) -> pd.DataFrame:
        """Run analysis on all storages and return results DataFrame"""
        results = []

        for model_type, config in self.storage_configs.items():
            dim = config["dim"]

            for storage_name in config["storages"]:
                storage_path = self.storage_base / f"{model_type}_{storage_name}"

                if not storage_path.exists():
                    logging.warning(f"Storage not found: {storage_path}")
                    continue

                metrics = self.analyze_storage(storage_path, dim)

                if metrics is not None:
                    metrics["model_type"] = model_type
                    metrics["dataset"] = storage_name
                    results.append(metrics)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Reorder columns for better readability
        column_order = [
            "model_type",
            "dataset",
            "total_embeddings",
            "successful_embeddings",
            "success_rate",
        ] + [
            col
            for col in df.columns
            if col
            not in [
                "model_type",
                "dataset",
                "total_embeddings",
                "successful_embeddings",
                "success_rate",
            ]
        ]

        return df[column_order]


def main():
    """Main execution function"""
    logging.info("Starting embedding metrics analysis")

    # Run analysis
    analyzer = StorageMetricsAnalyzer()
    results_df = analyzer.run_analysis()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/embedding_metrics_comparison_new_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)

    # Print summary
    logging.info("\nAnalysis Summary:")
    logging.info(f"Results saved to: {output_path}")

    # Display summary statistics
    print("\nResults Summary:")
    print(results_df.to_string())

    return results_df


# %%
if __name__ == "__main__":
    results = main()
