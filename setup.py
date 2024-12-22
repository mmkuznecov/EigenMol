from setuptools import setup, find_packages

setup(
    name="eigenmol",
    version="0.1.0",
    description="Molecular embedding generation using various models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "rdkit",
        "mol2vec",
        "gensim",
        "tqdm",
        "psutil"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'generate_molformer=scripts.molformer_emb_generation_dgsm:main',
            'generate_mol2vec=scripts.m2v_emb_generation_dgsm:main',
        ],
    }
)