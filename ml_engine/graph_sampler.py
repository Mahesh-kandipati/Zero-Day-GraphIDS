"""
Graph Sampler and Mini-Batch Loader.
Enables memory-efficient training over multiple graph chunks on CPU/GPU.
"""

from typing import List, Iterator, Dict, Any
import torch

try:
    from torch_geometric.data import Batch as PyGBatch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    class PyGBatch:
        @staticmethod
        def from_data_list(data_list):
            return data_list

class GraphBatchSampler:
    """
    Mini-batch loader for collections of graph chunks.
    """
    def __init__(self, chunks: List[Dict[str, Any]], batch_size: int = 16, shuffle: bool = True):
        self.chunks = chunks
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.chunks) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        n = len(self.chunks)
        indices = list(range(n))
        if self.shuffle:
            indices = torch.randperm(n).tolist()

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_chunks = [self.chunks[i] for i in indices[start:end]]
            yield batch_chunks

def create_pyg_batch(chunks: List[Dict[str, Any]]):
    """Combines a list of chunk data objects into a PyG Batch."""
    data_list = [c['pyg_data'] for c in chunks]
    if PYG_AVAILABLE:
        try:
            return PyGBatch.from_data_list(data_list)
        except Exception:
            return data_list
    return data_list
