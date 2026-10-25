"""
Graph Embedding Extraction and Dimensionality Reduction.
Generates latent representations from trained GNN encoders and computes
2D projections for interactive SOC dashboard visualization.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from sklearn.decomposition import PCA

def extract_chunk_embeddings(
    encoder: torch.nn.Module,
    chunks: List[Dict[str, Any]],
    device: str = 'cpu'
) -> Tuple[np.ndarray, List[int]]:
    """
    Passes each graph chunk through the trained GNN encoder to produce
    latent representation vectors for downstream anomaly scoring.
    """
    encoder.eval()
    embeddings = []
    chunk_ids = []

    with torch.no_grad():
        for chunk in chunks:
            pyg_data = chunk['pyg_data']
            x = pyg_data.x.to(device)
            edge_index = pyg_data.edge_index.to(device)

            if x.size(0) == 0:
                continue

            _, graph_embed, _ = encoder(x, edge_index)
            embed_vec = graph_embed.cpu().numpy().squeeze(0)
            embeddings.append(embed_vec)
            chunk_ids.append(chunk['chunk_id'])

    if not embeddings:
        return np.empty((0, encoder.graph_dim)), []

    return np.array(embeddings, dtype=np.float32), chunk_ids

def compute_2d_projections(
    embeddings: np.ndarray,
    labels: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Projects multi-dimensional graph embeddings into 2D space via PCA
    for scatter plot visualization on the SOC dashboard.
    """
    if len(embeddings) < 2:
        return []

    pca = PCA(n_components=2, random_state=42)
    proj_2d = pca.fit_transform(embeddings)

    points = []
    for i in range(len(proj_2d)):
        lbl = labels[i] if labels and i < len(labels) else "Normal"
        points.append({
            'chunk_id': i,
            'x': round(float(proj_2d[i, 0]), 4),
            'y': round(float(proj_2d[i, 1]), 4),
            'label': str(lbl)
        })

    return points
