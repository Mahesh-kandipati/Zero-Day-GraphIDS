"""
Graph Neural Network Encoders for Zero-Day GraphIDS.
Implements GraphSAGE and GCN architectures with multi-layer message passing,
graph pooling readouts, and reconstruction projection heads.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool, global_max_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

    class SAGEConv(nn.Module):
        """Native PyTorch GraphSAGE Mean-Aggregation Layer fallback."""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
            self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            n_nodes = x.size(0)
            if edge_index.numel() == 0 or n_nodes == 0:
                return self.lin_l(x)

            src, dst = edge_index[0], edge_index[1]
            # Aggregate neighbor features via scatter mean
            agg = torch.zeros_like(x)
            deg = torch.zeros(n_nodes, 1, device=x.device, dtype=x.dtype)
            agg.index_add_(0, dst, x[src])
            deg.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype))
            deg = deg.clamp(min=1.0)
            agg = agg / deg

            out = self.lin_l(x) + self.lin_r(agg)
            return out

    class GCNConv(SAGEConv):
        pass

    def global_mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones(batch.size(0), 1, device=x.device, dtype=x.dtype))
        return out / counts.clamp(min=1.0)

    def global_max_pool(x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return x.max(dim=0, keepdim=True)[0]
        num_graphs = int(batch.max().item()) + 1
        out = torch.full((num_graphs, x.size(1)), -1e9, device=x.device, dtype=x.dtype)
        for i in range(x.size(0)):
            b = batch[i].item()
            out[b] = torch.maximum(out[b], x[i])
        return out

class GraphSAGEEncoder(nn.Module):
    """
    Self-Supervised GraphSAGE Encoder Network.
    Maps flow-feature graph topologies to low-dimensional latent embeddings
    and reconstructs node features for unsupervised anomaly scoring.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        # Message Passing Layers
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Graph Readout & Projection Head (for Contrastive Learning / DGI)
        # Graph embedding combines mean and max pool (dimension: 2 * embedding_dim)
        self.graph_dim = embedding_dim * 2
        self.proj_head = nn.Sequential(
            nn.Linear(self.graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Node Feature Reconstruction Decoder (for Feature/Structure Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GraphSAGE message passing layers.
        Returns:
            node_embeds: [N, embedding_dim] - node level representations
            graph_embed: [B, graph_dim] - pooled graph level representation
            x_recon: [N, in_dim] - reconstructed original node features
        """
        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2
        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2)
        node_embeds = F.relu(h2)

        # Graph-level pooling readout (mean + max)
        h_mean = global_mean_pool(node_embeds, batch)
        h_max = global_max_pool(node_embeds, batch)
        graph_embed = torch.cat([h_mean, h_max], dim=-1)

        # Reconstructed node features
        x_recon = self.decoder(node_embeds)

        return node_embeds, graph_embed, x_recon

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method to retrieve graph embedding directly."""
        with torch.no_grad():
            _, graph_embed, _ = self.forward(x, edge_index, batch)
        return graph_embed
