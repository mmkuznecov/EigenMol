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
        "psutil",
        "click"  # Added for CLI support
    ],
    entry_points={
        'console_scripts': [
            'generate-m2v-dgsm=scripts.m2v_emb_generation_dgsm:main',
            'generate-m2v-peptides=scripts.m2v_emb_generation_peptides:main',
            'generate-molformer-dgsm=scripts.molformer_emb_generation_dgsm:main',
            'generate-molformer-peptides=scripts.molformer_emb_generation_peptides:main',
            'process-dgsm=scripts.process_dgsm:main',
            'process-peptides=scripts.process_peptides:main',
        ],
    },
    python_requires=">=3.8",
)
