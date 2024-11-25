import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor


def read_smi_file(file_path, num_samples=None):
    """
    Read SMI file and extract molecules

    Args:
        file_path (str): Path to the SMI file
        num_samples (int, optional): Number of samples to read. If None, reads all.

    Returns:
        list: List of molecules
    """
    # Read all lines from file
    with open(file_path) as file_handler:
        data = file_handler.readlines()

    # If num_samples specified, slice the data
    if num_samples is not None:
        data = data[:num_samples]

    # Extract molecules using parallel processing
    def return_molecule(molecule):
        return molecule.split("\t")[0]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        molecules = list(pool.map(return_molecule, data))

    return molecules


def analyze_molecules(molecules):
    """
    Analyze molecule lengths and create visualization

    Args:
        molecules (list): List of molecules

    Returns:
        tuple: (max_length, length_frequencies)
    """
    # Find maximum length
    max_length = len(max(molecules, key=len))
    print(f"Maximum length molecule is: {max_length}")

    # Get length frequencies
    len_mole = [len(mole) for mole in molecules]

    return max_length, len_mole


def save_to_parquet(molecules, output_path):
    """
    Save molecules to parquet format

    Args:
        molecules (list): List of molecules
        output_path (str): Path to save parquet file
    """
    df = pd.DataFrame(molecules, columns=["SMILES"])
    df.to_parquet(output_path, index=False)
    return df


def main():

    # Example usage:
    smi_file = "datasets/dgsm/chembl_22_clean_1576904_sorted_std_final.smi"
    parquet_file = "datasets/dgsm/chembl_22_clean_1576904_sorted_std_final.parquet"

    try:
        # Read molecules
        print("Reading molecules...")
        molecules = read_smi_file(smi_file)

        # Analyze data
        print("\nAnalyzing molecule lengths...")
        max_length, length_dist = analyze_molecules(molecules)

        # Save to parquet
        print("\nSaving to parquet format...")
        df = save_to_parquet(molecules, parquet_file)

        # Display some statistics
        print("\nDataset statistics:")
        print(f"Total number of molecules: {len(molecules)}")
        print(f"Maximum molecule length: {max_length}")
        print("\nFirst few molecules:")
        print(df.head())

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
