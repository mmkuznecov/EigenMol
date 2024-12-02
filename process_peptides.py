import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_single_smi_file(file_path):
    """Read single .smi file and return SMILES string"""
    try:
        with open(file_path) as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None


def process_simple_peptides(base_dir, peptide_type):
    """Process dipeptides and tripeptides"""
    smi_dir = Path(base_dir) / peptide_type / "smi"
    if not smi_dir.exists():
        raise ValueError(f"Directory not found: {smi_dir}")

    smiles_list = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        smi_files = list(smi_dir.glob("*.smi"))
        results = executor.map(read_single_smi_file, smi_files)

        for result in results:
            if result:
                smiles_list.append(result)

    df = pd.DataFrame({"SMILES": smiles_list})

    output_path = Path(base_dir) / peptide_type / f"{peptide_type}.parquet"
    df.to_parquet(output_path, index=False)

    logging.info(f"Processed {peptide_type}:")
    logging.info(f"Total molecules: {len(df)}")
    logging.info(f"Saved to: {output_path}")

    return df


def process_tetrapeptides(base_dir):
    """Process tetrapeptides with subdirectories"""
    smi_dir = Path(base_dir) / "tetrapeptides" / "smi"
    if not smi_dir.exists():
        raise ValueError(f"Directory not found: {smi_dir}")

    data = []

    # Process each subdirectory
    for subdir in smi_dir.iterdir():
        if subdir.is_dir():
            subdir_name = subdir.name

            # Process .smi files in subdirectory
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                smi_files = list(subdir.glob("*.smi"))
                results = executor.map(read_single_smi_file, smi_files)

                for result in results:
                    if result:
                        data.append({"SMILES": result, "subdir": subdir_name})

    df = pd.DataFrame(data)

    output_path = Path(base_dir) / "tetrapeptides" / "tetrapeptides.parquet"
    df.to_parquet(output_path, index=False)

    logging.info("Processed tetrapeptides:")
    logging.info(f"Total molecules: {len(df)}")
    logging.info(f"Unique subdirs: {len(df['subdir'].unique())}")
    logging.info(f"Saved to: {output_path}")

    return df


def main():
    base_dir = "data/peptides"

    try:
        # Process dipeptides and tripeptides
        for peptide_type in ["dipeptides", "tripeptides"]:
            df = process_simple_peptides(base_dir, peptide_type)
            logging.info(f"\nFirst few {peptide_type}:")
            logging.info(df.head())

        # Process tetrapeptides
        df_tetra = process_tetrapeptides(base_dir)
        logging.info("\nFirst few tetrapeptides:")
        logging.info(df_tetra.head())

        logging.info("\nProcessing completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
