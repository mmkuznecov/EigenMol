import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class MolformerEncoder:
    """
    A class to convert SMILES strings to molecular embeddings using MoLFormer
    """
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        """
        Initialize the encoder with MoLFormer model and tokenizer
        
        Args:
            model_name (str): Name or path of the MoLFormer model
        """
        try:
            print("Loading MoLFormer model and tokenizer...")
            self.model = AutoModel.from_pretrained(
                model_name, 
                deterministic_eval=True, 
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.model.eval()  # Set model to evaluation mode
            print("Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def get_embedding(self, smiles, pooling='pooler'):
        """
        Convert single SMILES string to embedding
        
        Args:
            smiles (str): SMILES string
            pooling (str): Pooling method ('pooler' or 'mean')
            
        Returns:
            numpy.ndarray: Molecular embedding
        """
        try:
            # Tokenize
            inputs = self.tokenizer([smiles], padding=True, return_tensors="pt", truncation=True)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            if pooling == 'pooler':
                # Use pooler_output (shape: [1, 768])
                embedding = outputs['pooler_output'].cpu().numpy()
            else:
                # Use mean of last_hidden_state (shape: [1, seq_len, 768])
                embedding = outputs['last_hidden_state'].mean(dim=1).cpu().numpy()
            
            return embedding[0]  # Return the embedding vector for single SMILES
            
        except Exception as e:
            print(f"Error processing SMILES string: {str(e)}")
            return None
            
    def batch_embed(self, smiles_list, batch_size=32, pooling='pooler'):
        """
        Convert a list of SMILES strings to embeddings in batches
        
        Args:
            smiles_list (list): List of SMILES strings
            batch_size (int): Batch size for processing
            pooling (str): Pooling method ('pooler' or 'mean')
            
        Returns:
            numpy.ndarray: Array of molecular embeddings
        """
        embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(batch, padding=True, return_tensors="pt", truncation=True)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                if pooling == 'pooler':
                    batch_embeddings = outputs['pooler_output'].cpu().numpy()
                else:
                    batch_embeddings = outputs['last_hidden_state'].mean(dim=1).cpu().numpy()
                
                embeddings.append(batch_embeddings)
                
            return np.vstack(embeddings)
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return None
    
    def process_dataframe(self, df, smiles_column='SMILES', batch_size=32, pooling='pooler'):
        """
        Process DataFrame containing SMILES strings
        
        Args:
            df (pd.DataFrame): Input DataFrame
            smiles_column (str): Name of column containing SMILES
            batch_size (int): Batch size for processing
            pooling (str): Pooling method ('pooler' or 'mean')
            
        Returns:
            pd.DataFrame: DataFrame with added embeddings
        """
        try:
            # Get embeddings for all SMILES
            embeddings = self.batch_embed(
                df[smiles_column].tolist(),
                batch_size=batch_size,
                pooling=pooling
            )
            
            # Add embeddings to DataFrame
            df['molformer_embedding'] = list(embeddings)
            
            return df
            
        except Exception as e:
            print(f"Error processing DataFrame: {str(e)}")
            return None