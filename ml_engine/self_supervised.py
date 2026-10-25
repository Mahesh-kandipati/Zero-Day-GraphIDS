"""
Self-Supervised Learning Objectives for Zero-Day GraphIDS.
Implements Deep Graph Infomax (DGI) Contrastive Mutual Information Maximization
and Masked Graph Feature Reconstruction without relying on attack labels.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearDiscriminator(nn.Module):
    """
    Discriminator scoring local-global mutual information between
    node-level embeddings and the global graph summary representation.
    """
    def __init__(self, node_dim: int, graph_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(node_dim, graph_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h_nodes: torch.Tensor, s_graph: torch.Tensor) -> torch.Tensor:
        """
        Calculates logit scores D(h_i, s) = h_i^T * W * s
        """
        # s_graph: [1, graph_dim], h_nodes: [N, node_dim]
        # Transformed graph vector: [graph_dim, 1] -> [node_dim, 1]
        s_trans = torch.matmul(self.weight, s_graph.t())  # [node_dim, 1]
        scores = torch.matmul(h_nodes, s_trans).squeeze(-1)  # [N]
        return scores

class SelfSupervisedGraphLoss(nn.Module):
    """
    Composite Self-Supervised Objective:
    1. Feature Reconstruction Loss: ||X - X_hat||^2
    2. Local-to-Global Contrastive InfoNCE / DGI Loss
    """
    def __init__(self, node_dim: int, graph_dim: int, contrastive_weight: float = 0.5):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.discriminator = BilinearDiscriminator(node_dim, graph_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def generate_corrupted_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a corrupted negative graph by randomly permuting node features
        across the batch while preserving graph topology.
        """
        perm_indices = torch.randperm(x.size(0), device=x.device)
        return x[perm_indices]

    def forward(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executes one self-supervised step over positive and negative views.
        """
        # 1. Positive Forward Pass
        node_embeds, graph_embed, x_recon = encoder(x, edge_index, batch)

        # Feature Reconstruction Loss
        recon_loss = self.mse_loss(x_recon, x)

        # 2. Corrupted Negative Graph Forward Pass
        x_corrupted = self.generate_corrupted_graph(x)
        neg_node_embeds, _, _ = encoder(x_corrupted, edge_index, batch)

        # 3. Contrastive Mutual Information Maximization (DGI)
        # Summary vector: summary = sigmoid(graph_embed)
        summary = torch.sigmoid(graph_embed)

        pos_scores = self.discriminator(node_embeds, summary)
        neg_scores = self.discriminator(neg_node_embeds, summary)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        loss_pos = self.bce_loss(pos_scores, pos_labels)
        loss_neg = self.bce_loss(neg_scores, neg_labels)
        contrastive_loss = 0.5 * (loss_pos + loss_neg)

        # Total Composite Loss
        total_loss = recon_loss + (self.contrastive_weight * contrastive_loss)

        metrics = {
            'total_loss': float(total_loss.item()),
            'recon_loss': float(recon_loss.item()),
            'contrastive_loss': float(contrastive_loss.item())
        }

        return total_loss, metrics
