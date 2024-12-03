import os
import numpy as np
from typing import List, Iterator, Dict, Any, Optional, Union, Tuple
import json
import logging
from tqdm import tqdm
import pandas as pd
import shutil

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TensorStorage:
    """
    TensorStorage that stores tensor data in chunks and metadata in parquet.
    Provides mapping between tensor indices and additional metadata parameters.
    """

    def __init__(
        self,
        storage_dir: str,
        description: str = "",
        chunk_size: Optional[int] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize the TensorStorage.

        Args:
            storage_dir (str): Directory where the storage will be created or loaded from.
            description (str): Optional description of the storage
            chunk_size (Optional[int]): Size of each chunk in bytes. If None, will be loaded from metadata
                                      or set to default value.
            return_metadata (bool): If True, __getitem__ will return (tensor, metadata) pairs
        """
        self.storage_dir = storage_dir
        self.description = description
        self.return_metadata = return_metadata
        self.chunks_dir = os.path.join(storage_dir, "chunks")
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        self.metadata_file = os.path.join(self.metadata_dir, "metadata.json")
        self.parquet_file = os.path.join(self.metadata_dir, "tensor_metadata.parquet")

        # Create directories if they don't exist
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.metadata = self._load_metadata()

        # Set chunk size with priority: provided > metadata > default
        default_chunk_size = 3 * 2**20 * np.dtype(np.float32).itemsize
        self.chunk_size = chunk_size or self.metadata.get(
            "chunk_size", default_chunk_size
        )

        self.loaded_chunks = {}
        self.current_window = []

        # Load parquet metadata
        self.metadata_df = self.load_metadata_table()

    def _load_parquet_metadata(self):
        """Load the parquet metadata if it exists."""
        if os.path.exists(self.parquet_file):
            self.metadata_df = pd.read_parquet(self.parquet_file)
        else:
            self.metadata_df = pd.DataFrame()

    def _load_metadata(self):
        """Load metadata from the JSON file if it exists, otherwise return an empty dict."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save the current metadata to the JSON file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def _get_chunk_filename(self, chunk_id: int) -> str:
        """Generate the filename for a given chunk ID."""
        return os.path.join(self.chunks_dir, f"ch{chunk_id}")

    def _load_chunk(self, chunk_id: int) -> np.ndarray:
        """
        Load a chunk into memory if it's not already loaded.

        Args:
            chunk_id (int): ID of the chunk to load.

        Returns:
            np.ndarray: The loaded chunk data.
        """
        if chunk_id not in self.loaded_chunks:
            chunk_file = self._get_chunk_filename(chunk_id)
            self.loaded_chunks[chunk_id] = np.fromfile(chunk_file, dtype=np.float32)
        return self.loaded_chunks[chunk_id]

    def _unload_chunk(self, chunk_id: int):
        """Remove a chunk from memory."""
        if chunk_id in self.loaded_chunks:
            del self.loaded_chunks[chunk_id]

    def _update_window(self, needed_chunks: List[int]):
        """
        Update the window of loaded chunks based on what's needed.
        Unload unnecessary chunks and load new ones.

        Args:
            needed_chunks (List[int]): List of chunk IDs that are needed.
        """
        new_window = set(needed_chunks)
        for chunk_id in self.current_window:
            if chunk_id not in new_window:
                self._unload_chunk(chunk_id)
        for chunk_id in new_window:
            if chunk_id not in self.current_window:
                self._load_chunk(chunk_id)
        self.current_window = list(new_window)

    def __getitem__(
        self, idx: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a tensor from storage by its index.

        Args:
            idx (int): Index of the tensor to retrieve.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
                If return_metadata is False: just the tensor
                If return_metadata is True: tuple of (tensor, metadata_dict)

        Raises:
            IndexError: If the index is not found in storage.
        """
        if str(idx) not in self.metadata["elements"]:
            raise IndexError(f"Index {idx} not found in storage")

        item_meta = self.metadata["elements"][str(idx)]
        chunks_info = item_meta["chunks"]
        shape = item_meta["shape"]

        needed_chunks = [chunk_info[0] for chunk_info in chunks_info]
        self._update_window(needed_chunks)

        data = []
        for chunk_id, start_idx, end_idx in chunks_info:
            chunk_data = self._load_chunk(chunk_id)[start_idx:end_idx]
            data.append(chunk_data)

        tensor = np.concatenate(data).reshape(shape)

        if not self.return_metadata:
            return tensor

        # Get metadata if requested
        if self.metadata_df is not None and not self.metadata_df.empty:
            metadata = (
                self.metadata_df[self.metadata_df["tensor_idx"] == idx]
                .iloc[0]
                .to_dict()
            )
        else:
            metadata = {}

        return tensor, metadata

    def __len__(self):
        """Return the number of tensors in the storage."""
        return len(self.metadata["elements"])

    def __repr__(self) -> str:
        """Return string representation of the storage."""
        total_size = 0
        if os.path.exists(self.chunks_dir):
            for chunk_file in os.listdir(self.chunks_dir):
                total_size += os.path.getsize(os.path.join(self.chunks_dir, chunk_file))

        info = [
            f"TensorStorage at '{self.storage_dir}'",
            f"Description: {self.description}",
            f"Number of tensors: {len(self)}",
            f"Chunk size: {self.chunk_size / (1024 * 1024):.2f} MB",
            f"Total storage size: {total_size / (1024 * 1024):.2f} MB",
        ]

        # Add shape information if available
        if len(self) > 0:
            first_tensor_meta = self.metadata["elements"]["0"]
            info.append(f"Tensor shape: {first_tensor_meta['shape']}")

        return "\n".join(info)

    def load_metadata_table(self) -> Optional[pd.DataFrame]:
        """
        Load the metadata table from parquet file.

        Returns:
            Optional[pd.DataFrame]: DataFrame with metadata or None if file doesn't exist
        """
        if os.path.exists(self.parquet_file):
            return pd.read_parquet(self.parquet_file)
        return None

    def get_storage_info(self) -> Dict[str, Any]:
        """Get detailed information about the storage."""
        total_size = 0
        if os.path.exists(self.chunks_dir):
            for chunk_file in os.listdir(self.chunks_dir):
                total_size += os.path.getsize(os.path.join(self.chunks_dir, chunk_file))

        info = {
            "storage_dir": self.storage_dir,
            "description": self.description,
            "num_tensors": len(self),
            "chunk_size_mb": self.chunk_size / (1024 * 1024),
            "total_size_mb": total_size / (1024 * 1024),
        }

        if len(self) > 0:
            first_tensor_meta = self.metadata["elements"]["0"]
            info["tensor_shape"] = first_tensor_meta["shape"]

        return info

    def get_tensor_by_param(
        self, param_name: str, param_value: Any
    ) -> Optional[np.ndarray]:
        """
        Retrieve a tensor by querying a parameter in the metadata.

        Args:
            param_name (str): Name of the parameter to query
            param_value (Any): Value to search for

        Returns:
            Optional[np.ndarray]: The tensor if found, None otherwise
        """
        matches = self.metadata_df[self.metadata_df[param_name] == param_value]
        if len(matches) == 0:
            return None

        tensor_idx = matches.iloc[0]["tensor_idx"]
        return self[tensor_idx]

    def get_params_for_tensor(self, tensor_idx: int) -> Dict[str, Any]:
        """
        Get all parameters associated with a tensor.

        Args:
            tensor_idx (int): Index of the tensor

        Returns:
            Dict[str, Any]: Dictionary of parameters
        """
        matches = self.metadata_df[self.metadata_df["tensor_idx"] == tensor_idx]
        if len(matches) == 0:
            return {}

        return matches.iloc[0].to_dict()

    def get_tensors_by_batch(self, batch_id: int) -> List[np.ndarray]:
        """
        Retrieve all tensors associated with a specific batch ID.

        Args:
            batch_id (int): The batch ID to query

        Returns:
            List[np.ndarray]: List of tensors in the batch
        """
        matches = self.metadata_df[self.metadata_df["batch_id"] == batch_id]
        return [self[idx] for idx in matches["tensor_idx"]]

    def filter_tensors(self, **kwargs) -> List[int]:
        """
        Filter tensors based on multiple metadata parameters.

        Args:
            **kwargs: Key-value pairs of metadata parameters to filter by

        Returns:
            List[int]: List of tensor indices matching all criteria
        """
        filtered_df = self.metadata_df
        for key, value in kwargs.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        return filtered_df["tensor_idx"].tolist()

    @staticmethod
    def create_storage(
        storage_dir: str,
        data_iterator: Iterator[np.ndarray],
        metadata_iterator: Iterator[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        description: str = "",
    ) -> "TensorStorage":
        """
        Create a new TensorStorage from iterators of numpy arrays and metadata.

        Args:
            storage_dir (str): Directory where the storage will be created.
            data_iterator (Iterator[np.ndarray]): Iterator yielding numpy arrays to store.
            metadata_iterator (Iterator[Dict[str, Any]]): Iterator yielding metadata dicts.
            chunk_size (int): Size of each chunk in bytes.
            description (str): Optional description of the storage.

        Returns:
            TensorStorage: The created storage instance.
        """
        os.makedirs(os.path.join(storage_dir, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "metadata"), exist_ok=True)

        storage = TensorStorage(storage_dir, description, chunk_size)

        logging.info(f"Creating storage in directory: {storage_dir}")
        logging.info(f"Chunk size: {storage.chunk_size / (1024 * 1024):.2f} MB")

        current_chunk = []
        current_chunk_size = 0
        chunk_id = 0
        elements_metadata = {}
        metadata_records = []
        total_elements = 0

        progress_bar = tqdm(desc="Processing arrays", unit="array")

        for idx, (arr, metadata_dict) in enumerate(
            zip(data_iterator, metadata_iterator)
        ):
            total_elements += 1
            progress_bar.update(1)

            flat_arr = arr.flatten()
            arr_size = flat_arr.nbytes

            if arr_size > storage.chunk_size:
                raise ValueError(
                    f"Array at index {idx} is larger than the maximum chunk size"
                )

            # Handle chunk storage
            if current_chunk_size + arr_size > storage.chunk_size:
                chunk_filename = storage._get_chunk_filename(chunk_id)
                np.concatenate(current_chunk).astype(np.float32).tofile(chunk_filename)
                chunk_id += 1
                current_chunk = []
                current_chunk_size = 0

            start_idx = current_chunk_size // np.dtype(np.float32).itemsize
            end_idx = start_idx + flat_arr.size
            current_chunk.append(flat_arr)
            current_chunk_size += arr_size

            # Store tensor metadata
            elements_metadata[str(idx)] = {
                "shape": arr.shape,
                "chunks": [(chunk_id, start_idx, end_idx)],
            }

            # Store additional metadata
            metadata_dict["tensor_idx"] = idx
            metadata_records.append(metadata_dict)

        # Save the last chunk if there's any data left
        if current_chunk:
            chunk_filename = storage._get_chunk_filename(chunk_id)
            np.concatenate(current_chunk).astype(np.float32).tofile(chunk_filename)

        progress_bar.close()

        # Save the tensor metadata
        storage.metadata = {
            "chunk_size": chunk_size,
            "total_elements": total_elements,
            "total_chunks": chunk_id + 1,
            "elements": elements_metadata,
            "description": description,
        }
        storage._save_metadata()

        # Save the parquet metadata
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(storage.parquet_file, index=False)
        storage.metadata_df = metadata_df

        logging.info(f"Storage creation complete. Total elements: {total_elements}")
        logging.info(f"Total chunks created: {chunk_id + 1}")
        return storage

    def close(self):
        """Clean up resources and ensure all metadata is saved."""
        self._save_metadata()
        self.loaded_chunks.clear()
        self.current_window.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def rebuild_storage(
        self,
        new_storage_dir: Optional[str] = None,
        new_chunk_size: Optional[int] = None,
        description: Optional[str] = None,
        inplace: bool = False
    ) -> "TensorStorage":
        """
        Rebuild the storage with new chunk size.
        
        Args:
            new_storage_dir (Optional[str]): Directory for the new storage. 
                                           If None and inplace=True, will rebuild in place.
            new_chunk_size (Optional[int]): New chunk size in bytes. If None, uses current chunk size
            description (Optional[str]): New description. If None, uses current description
            inplace (bool): If True, will rebuild the storage in place, replacing current storage
            
        Returns:
            TensorStorage: New storage instance with updated chunk size
            
        Raises:
            ValueError: If inplace is True and new_storage_dir is provided
        """
        if inplace and new_storage_dir is not None:
            raise ValueError("Cannot specify new_storage_dir when inplace=True")
            
        if inplace:
            # Create temporary directory for rebuilding
            temp_dir = os.path.join(os.path.dirname(self.storage_dir), f"{os.path.basename(self.storage_dir)}_temp")
        else:
            if new_storage_dir is None:
                raise ValueError("Must specify new_storage_dir when inplace=False")
            temp_dir = new_storage_dir
            
        # Use current values if new ones not provided
        new_chunk_size = new_chunk_size or self.chunk_size
        new_description = description or self.description
        
        try:
            # Create iterators for current data
            def tensor_iterator() -> Iterator[np.ndarray]:
                for i in range(len(self)):
                    yield self[i]
                    
            def metadata_iterator() -> Iterator[Dict[str, Any]]:
                for i in range(len(self)):
                    yield self.get_params_for_tensor(i)
            
            # Create new storage with different chunk size
            new_storage = TensorStorage.create_storage(
                storage_dir=temp_dir,
                data_iterator=tensor_iterator(),
                metadata_iterator=metadata_iterator(),
                chunk_size=new_chunk_size,
                description=new_description
            )
            
            if inplace:
                # Verify the new storage before replacing
                self._verify_rebuilt_storage(new_storage)
                
                # Close both storages to ensure all files are written
                self.close()
                new_storage.close()
                
                # Replace old storage with new one
                backup_dir = f"{self.storage_dir}_backup"
                os.rename(self.storage_dir, backup_dir)
                try:
                    os.rename(temp_dir, self.storage_dir)
                except Exception as e:
                    # If something goes wrong, restore from backup
                    os.rename(backup_dir, self.storage_dir)
                    raise e
                
                # Remove backup after successful replacement
                shutil.rmtree(backup_dir)
                
                # Reinitialize self with new storage
                self.__init__(
                    storage_dir=self.storage_dir,
                    description=new_description,
                    chunk_size=new_chunk_size,
                    return_metadata=self.return_metadata
                )
                return self
            else:
                return new_storage
                
        except Exception as e:
            # Clean up temporary directory if something goes wrong
            if inplace and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e

    def _verify_rebuilt_storage(self, new_storage: "TensorStorage") -> bool:
        """
        Verify that the rebuilt storage contains the same data.
        
        Args:
            new_storage: The newly built storage to verify
            
        Returns:
            bool: True if verification passes
            
        Raises:
            ValueError: If verification fails
        """
        # Verify basic properties
        if len(self) != len(new_storage):
            raise ValueError("New storage has different number of elements")
            
        # Verify sample of tensors
        sample_size = min(100, len(self))  # Check up to 100 random tensors
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        for idx in indices:
            original_tensor = self[idx]
            new_tensor = new_storage[idx]
            if not np.allclose(original_tensor, new_tensor):
                raise ValueError(f"Data mismatch at index {idx}")
                
            original_meta = self.get_params_for_tensor(idx)
            new_meta = new_storage.get_params_for_tensor(idx)
            if original_meta != new_meta:
                raise ValueError(f"Metadata mismatch at index {idx}")
                
        return True
