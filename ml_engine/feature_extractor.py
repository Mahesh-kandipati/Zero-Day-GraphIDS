"""
Feature Extraction and Graph Topology Summarization.
Extracts normalized node feature representations and structural summary statistics.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

class NetworkFeatureExtractor:
    """
    Extracts, scales, and prepares tabular network flow records
    for graph construction and neural network ingestion.
    """
    def __init__(self, scaler: StandardScaler = None):
        self.scaler = scaler or StandardScaler()
        self.is_fitted = scaler is not None
        self.selected_features: List[str] = []

    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """Fit scaler on normal baseline network flow features."""
        self.scaler.fit(X)
        self.is_fitted = True
        self.selected_features = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform network features using fitted scaler with infinite/nan protection."""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        if not self.is_fitted:
            return self.scaler.fit_transform(X_clean)
        return self.scaler.transform(X_clean)

    def fit_transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    @staticmethod
    def extract_graph_structural_features(G: nx.Graph, node_features: np.ndarray) -> np.ndarray:
        """
        Combines statistical distributions of flow features with graph topological metrics:
        1. Feature-wise means and standard deviations across flows in the graph chunk.
        2. Mean node degree and degree standard deviation.
        3. Mean clustering coefficient and graph density.
        Returns a 1D graph-level summary vector.
        """
        n_nodes = G.number_of_nodes()
        if n_nodes == 0 or node_features.shape[0] == 0:
            return np.zeros(node_features.shape[1] * 2 + 4, dtype=np.float32)

        # Statistical features across flows in chunk
        feat_means = np.mean(node_features, axis=0)
        feat_stds = np.std(node_features, axis=0)

        # Topological features from NetworkX graph
        degrees = [d for _, d in G.degree()]
        mean_deg = float(np.mean(degrees)) if degrees else 0.0
        std_deg = float(np.std(degrees)) if degrees else 0.0

        if n_nodes > 1:
            try:
                clustering = float(nx.average_clustering(G))
            except Exception:
                clustering = 0.0
            density = float(nx.density(G))
        else:
            clustering = 0.0
            density = 0.0

        topo_vec = np.array([mean_deg, std_deg, clustering, density], dtype=np.float32)

        # Concatenate: [mean_features, std_features, topological_features]
        summary_vec = np.concatenate([feat_means, feat_stds, topo_vec]).astype(np.float32)
        return np.nan_to_num(summary_vec, nan=0.0, posinf=1.0, neginf=-1.0)
