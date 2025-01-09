import click
import pandas as pd
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def read_smi_file(file_path: str, num_samples: Optional[int] = None) -> List[str]:
    """
    Read SMI file and extract molecules.

    Args:
        file_path (str): Path to the SMI file
        num_samples (int, optional): Number of samples to read. If None, reads all.

    Returns:
        list: List of SMILES strings
    """
    logging.info(f"Reading SMI file: {file_path}")

    try:
        # Read all lines from file
        with open(file_path) as file_handler:
            data = file_handler.readlines()

        # If num_samples specified, slice the data
        if num_samples is not None:
            data = data[:num_samples]
            logging.info(f"Reading {num_samples} samples")

        # Extract molecules using parallel processing
        def return_molecule(molecule: str) -> str:
            try:
                return molecule.split("\t")[0].strip()
            except Exception as e:
                logging.warning(f"Error processing molecule: {e}")
                return ""

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            molecules = list(
                tqdm(
                    pool.map(return_molecule, data),
                    total=len(data),
                    desc="Extracting SMILES",
                )
            )

        # Filter out empty entries
        molecules = [mol for mol in molecules if mol]

        logging.info(f"Successfully processed {len(molecules)} molecules")
        return molecules

    except Exception as e:
        logging.error(f"Error reading SMI file: {e}")
        raise


def analyze_molecules(molecules: List[str]) -> tuple:
    """
    Analyze molecule lengths and create statistics.

    Args:
        molecules (list): List of SMILES strings

    Returns:
        tuple: (max_length, length_frequencies)
    """
    try:
        # Find maximum length
        max_length = len(max(molecules, key=len))
        logging.info(f"Maximum SMILES length: {max_length}")

        # Get length frequencies
        len_mole = [len(mole) for mole in molecules]

        # Calculate statistics
        avg_length = sum(len_mole) / len(len_mole)
        min_length = min(len_mole)

        logging.info(f"Average SMILES length: {avg_length:.2f}")
        logging.info(f"Minimum SMILES length: {min_length}")

        return max_length, len_mole

    except Exception as e:
        logging.error(f"Error analyzing molecules: {e}")
        raise


def save_to_parquet(molecules: List[str], output_path: str) -> pd.DataFrame:
    """
    Save molecules to parquet format.

    Args:
        molecules (list): List of SMILES strings
        output_path (str): Path to save parquet file

    Returns:
        pd.DataFrame: DataFrame with saved data
    """
    try:
        # Create DataFrame
        df = pd.DataFrame(molecules, columns=["SMILES"])

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to parquet
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved {len(df)} molecules to: {output_path}")

        return df

    except Exception as e:
        logging.error(f"Error saving to parquet: {e}")
        raise


@click.command()
@click.option("--input-file", required=True, type=str, help="Path to input SMI file")
@click.option(
    "--output-file", required=True, type=str, help="Path for output parquet file"
)
@click.option("--num-samples", type=int, help="Number of samples to process (optional)")
@click.option("--log-level", default="INFO", help="Logging level")
def main(input_file: str, output_file: str, num_samples: Optional[int], log_level: str):
    """Process DGSM dataset from SMI to parquet format."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logging.info("Starting DGSM dataset processing")
        logging.info(f"Input file: {input_file}")
        logging.info(f"Output file: {output_file}")

        # Read molecules
        molecules = read_smi_file(input_file, num_samples)

        # Analyze data
        logging.info("\nAnalyzing molecule lengths...")
        max_length, length_dist = analyze_molecules(molecules)

        # Save to parquet
        logging.info("\nSaving to parquet format...")
        df = save_to_parquet(molecules, output_file)

        # Display summary statistics
        logging.info("\nDataset Summary:")
        logging.info(f"Total molecules processed: {len(molecules)}")
        logging.info(f"Maximum SMILES length: {max_length}")
        logging.info("\nFirst few molecules:")
        pd.set_option("display.max_colwidth", None)
        logging.info("\n" + str(df.head()))

        logging.info("\nProcessing completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/process_dgsm.py \
#     --input-file data/dgsm/chembl_22_clean_1576904_sorted_std_final.smi \
#     --output-file data/dgsm/chembl_22_clean_1576904_sorted_std_final.parquet \
#     --log-level INFO
