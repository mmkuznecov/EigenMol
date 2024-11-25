import numpy as np
import pandas as pd
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec, DfVec
from gensim.models import word2vec
import warnings
warnings.filterwarnings('ignore')

class Mol2VecEncoder:
    """
    A class to convert SMILES strings to molecular vector embeddings using mol2vec
    """
    def __init__(self, model_path):
        """
        Initialize the encoder with a pre-trained mol2vec model
        
        Args:
            model_path (str): Path to the pre-trained mol2vec model
        """
        try:
            self.model = word2vec.Word2Vec.load(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def smiles_to_mol(self, smiles):
        """
        Convert SMILES string to RDKit molecule with hydrogens
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            rdkit.Chem.rdchem.Mol: RDKit molecule object
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            mol = Chem.AddHs(mol)
            return mol
        except Exception as e:
            print(f"Error converting SMILES to molecule: {str(e)}")
            return None
            
    def get_mol_sentence(self, mol, radius=1):
        """
        Convert molecule to molecular sentence
        
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
            radius (int): Radius for morgan fingerprint
            
        Returns:
            mol2vec.features.MolSentence: Molecular sentence object
        """
        try:
            sentence = mol2alt_sentence(mol, radius)
            return MolSentence(sentence)
        except Exception as e:
            print(f"Error creating molecular sentence: {str(e)}")
            return None
            
    def get_embedding(self, mol_sentence):
        """
        Convert molecular sentence to vector embedding
        
        Args:
            mol_sentence (mol2vec.features.MolSentence): Molecular sentence object
            
        Returns:
            numpy.ndarray: Molecular vector embedding
        """
        try:
            vec = sentences2vec([mol_sentence], self.model, unseen='UNK')
            return DfVec(vec).vec
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return None
            
    def smiles_to_vec(self, smiles, radius=1):
        """
        Convert SMILES string to molecular vector embedding
        
        Args:
            smiles (str): SMILES string
            radius (int): Radius for morgan fingerprint
            
        Returns:
            numpy.ndarray: Molecular vector embedding
        """
        try:
            # Convert SMILES to molecule
            mol = self.smiles_to_mol(smiles)
            if mol is None:
                return None
                
            # Get molecular sentence
            mol_sentence = self.get_mol_sentence(mol, radius)
            if mol_sentence is None:
                return None
                
            # Get embedding
            embedding = self.get_embedding(mol_sentence)
            return embedding
            
        except Exception as e:
            print(f"Error processing SMILES string: {str(e)}")
            return None
            
    def batch_smiles_to_vec(self, smiles_list, radius=1):
        """
        Convert a list of SMILES strings to molecular vector embeddings
        
        Args:
            smiles_list (list): List of SMILES strings
            radius (int): Radius for morgan fingerprint
            
        Returns:
            numpy.ndarray: Array of molecular vector embeddings
        """
        embeddings = []
        for smiles in smiles_list:
            embedding = self.smiles_to_vec(smiles, radius)
            if embedding is not None:
                embeddings.append(embedding)
        return np.array(embeddings)