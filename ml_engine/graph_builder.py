"""
Graph Construction Engine for Zero-Day GraphIDS.
Constructs k-NN metric similarity graphs and host-flow communication graphs
from preprocessed tabular network traffic flows.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

try:
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    # Fallback container class matching PyG Data interface
    class PyGData:
        def __init__(self, x=None, edge_index=None, edge_attr=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            for k, v in kwargs.items():
                setattr(self, k, v)

class GraphBuilder:
    """
    Constructs PyTorch Geometric Data graphs and NetworkX topologies from flow records.
    """
    def __init__(self, k_neighbors: int = 5, metric: str = 'euclidean'):
        self.k_neighbors = max(1, k_neighbors)
        self.metric = metric

    def build_knn_graph(
        self,
        node_features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[PyGData, nx.Graph]:
        """
        Builds a k-Nearest Neighbors similarity graph for a batch of network flows.
        Nodes: individual network flows.
        Edges: metric similarity between flows in normalized feature space.
        """
        n_samples = node_features.shape[0]
        if n_samples == 0:
            empty_x = torch.empty((0, node_features.shape[1]), dtype=torch.float32)
            empty_edge = torch.empty((2, 0), dtype=torch.long)
            return PyGData(x=empty_x, edge_index=empty_edge), nx.Graph()

        # NetworkX graph representation for analytics and topological extraction
        G = nx.Graph()
        for i in range(n_samples):
            lbl = str(labels[i]) if labels is not None and i < len(labels) else "Normal"
            G.add_node(i, label=lbl)

        # For very small batches (n <= 1), create isolated node with self loop
        if n_samples <= 1:
            x_tensor = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.ones((1, 1), dtype=torch.float32)
            G.add_edge(0, 0, weight=1.0)
            data = PyGData(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
            return data, G

        effective_k = min(self.k_neighbors + 1, n_samples)
        edge_list_src = []
        edge_list_dst = []
        edge_weights = []

        if self.metric == 'cosine':
            # Pairwise cosine similarity
            sim_matrix = cosine_similarity(node_features)
            for i in range(n_samples):
                # Get top-k similar indices (excluding self)
                sim_scores = sim_matrix[i]
                top_indices = np.argsort(sim_scores)[::-1][1:effective_k]
                for j in top_indices:
                    edge_list_src.append(i)
                    edge_list_dst.append(j)
                    weight = float(max(0.0, sim_scores[j]))
                    edge_weights.append([weight])
                    G.add_edge(i, j, weight=weight)
        else:
            # Pairwise Euclidean k-NN
            nbrs = NearestNeighbors(n_neighbors=effective_k, metric='euclidean', algorithm='auto')
            nbrs.fit(node_features)
            distances, indices = nbrs.kneighbors(node_features)

            for i in range(n_samples):
                for rank, j in enumerate(indices[i][1:], start=1):
                    dist = distances[i][rank]
                    # Convert distance to normalized similarity weight: exp(-d^2 / 2)
                    weight = float(np.exp(- (dist ** 2) / 2.0))
                    edge_list_src.append(i)
                    edge_list_dst.append(j)
                    edge_weights.append([weight])
                    G.add_edge(i, j, weight=weight)

        # Add bidirectional edges if not already present
        edge_index_np = np.array([edge_list_src, edge_list_dst], dtype=np.int64)

        x_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_weights, dtype=torch.float32)

        data = PyGData(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            num_nodes=n_samples
        )
        return data, G

    def build_flow_chunks(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        chunk_size: int = 100,
        overlap_ratio: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Slices sequential network flow records into graph chunks.
        Each chunk is converted into a PyG Data graph and NetworkX graph.
        """
        n_total = len(features)
        if n_total == 0:
            return []

        step = max(1, int(chunk_size * (1.0 - overlap_ratio)))
        chunks = []

        chunk_id = 0
        for start_idx in range(0, n_total, step):
            end_idx = min(start_idx + chunk_size, n_total)
            chunk_features = features[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx] if labels is not None else None

            # Skip tiny tail chunk if less than 5 records and earlier chunks exist
            if len(chunk_features) < 5 and len(chunks) > 0:
                continue

            pyg_data, nx_graph = self.build_knn_graph(chunk_features, chunk_labels)

            chunk_info = {
                'chunk_id': chunk_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': len(chunk_features),
                'pyg_data': pyg_data,
                'nx_graph': nx_graph,
                'features': chunk_features,
                'labels': chunk_labels
            }
            chunks.append(chunk_info)
            chunk_id += 1

            if end_idx >= n_total:
                break

        return chunks
