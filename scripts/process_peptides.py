import click
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from typing import List, Dict, Optional
from tqdm import tqdm


def read_single_smi_file(file_path: Path) -> Optional[str]:
    """
    Read single .smi file and return SMILES string.

    Args:
        file_path (Path): Path to SMI file

    Returns:
        Optional[str]: SMILES string if successful, None otherwise
    """
    try:
        with open(file_path) as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None


def process_simple_peptides(base_dir: Path, peptide_type: str) -> pd.DataFrame:
    """
    Process dipeptides and tripeptides.

    Args:
        base_dir (Path): Base directory for peptide data
        peptide_type (str): Type of peptides ('dipeptides' or 'tripeptides')

    Returns:
        pd.DataFrame: Processed peptide data
    """
    smi_dir = base_dir / peptide_type / "smi"
    if not smi_dir.exists():
        raise ValueError(f"Directory not found: {smi_dir}")

    logging.info(f"\nProcessing {peptide_type}...")
    smiles_list = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        smi_files = list(smi_dir.glob("*.smi"))
        logging.info(f"Found {len(smi_files)} SMI files")

        # Process files with progress bar
        results = list(
            tqdm(
                executor.map(read_single_smi_file, smi_files),
                total=len(smi_files),
                desc=f"Processing {peptide_type}",
            )
        )

        smiles_list = [result for result in results if result]

    df = pd.DataFrame({"SMILES": smiles_list})

    # Create output directory and save
    output_dir = base_dir / peptide_type
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{peptide_type}.parquet"
    df.to_parquet(output_path, index=False)

    # Log statistics
    logging.info(f"Processed {peptide_type}:")
    logging.info(f"Total molecules: {len(df)}")
    logging.info(f"Saved to: {output_path}")

    return df


def process_tetrapeptides(base_dir: Path) -> pd.DataFrame:
    """
    Process tetrapeptides with subdirectories.

    Args:
        base_dir (Path): Base directory for peptide data

    Returns:
        pd.DataFrame: Processed tetrapeptide data
    """
    smi_dir = base_dir / "tetrapeptides" / "smi"
    if not smi_dir.exists():
        raise ValueError(f"Directory not found: {smi_dir}")

    logging.info("\nProcessing tetrapeptides...")
    data = []
    subdirs = list(smi_dir.iterdir())

    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        if subdir.is_dir():
            subdir_name = subdir.name
            logging.debug(f"Processing subdir: {subdir_name}")

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                smi_files = list(subdir.glob("*.smi"))
                results = list(executor.map(read_single_smi_file, smi_files))

                for result in results:
                    if result:
                        data.append({"SMILES": result, "subdir": subdir_name})

    df = pd.DataFrame(data)

    # Create output directory and save
    output_dir = base_dir / "tetrapeptides"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tetrapeptides.parquet"
    df.to_parquet(output_path, index=False)

    # Log statistics
    logging.info("Processed tetrapeptides:")
    logging.info(f"Total molecules: {len(df)}")
    logging.info(f"Unique subdirs: {len(df['subdir'].unique())}")
    logging.info(f"Saved to: {output_path}")

    return df


@click.command()
@click.option(
    "--input-dir", required=True, type=str, help="Base directory for input peptide data"
)
@click.option(
    "--peptide-type",
    type=click.Choice(["di", "tri", "tetra", "all"]),
    required=True,
    help="Type of peptides to process",
)
@click.option("--log-level", default="INFO", help="Logging level")
def main(input_dir: str, peptide_type: str, log_level: str):
    """Process peptide datasets from SMI to parquet format."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    base_dir = Path(input_dir)
    if not base_dir.exists():
        raise ValueError(f"Base directory not found: {base_dir}")

    try:
        # Process based on peptide type
        if peptide_type == "all":
            # Process all types
            for ptype in ["di", "tri", "tetra"]:
                if ptype in ["di", "tri"]:
                    peptide_name = "dipeptides" if ptype == "di" else "tripeptides"
                    df = process_simple_peptides(base_dir, peptide_name)
                    logging.info(f"\nFirst few {peptide_name}:")
                    logging.info(df.head())
                else:
                    df_tetra = process_tetrapeptides(base_dir)
                    logging.info("\nFirst few tetrapeptides:")
                    logging.info(df_tetra.head())
        else:
            # Process specific type
            if peptide_type in ["di", "tri"]:
                peptide_name = "dipeptides" if peptide_type == "di" else "tripeptides"
                df = process_simple_peptides(base_dir, peptide_name)
                logging.info(f"\nFirst few {peptide_name}:")
                logging.info(df.head())
            else:
                df = process_tetrapeptides(base_dir)
                logging.info("\nFirst few tetrapeptides:")
                logging.info(df.head())

        logging.info("\nProcessing completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()

# python scripts/process_peptides.py \
#     --input-dir data/peptides \
#     --peptide-type all \
#     --log-level INFO
