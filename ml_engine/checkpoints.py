"""
Model Checkpointing and Safe Artifact Serialization.
Persists trained GNN weights, feature scalers, calibration thresholds, and metadata.
"""

import os
from typing import Dict, Any, Optional
import torch
from .encoder import GraphSAGEEncoder

def save_checkpoint(
    file_path: str,
    encoder: GraphSAGEEncoder,
    scaler_params: Dict[str, Any],
    threshold_info: Dict[str, Any],
    metadata: Dict[str, Any],
    discriminator: Optional[torch.nn.Module] = None
) -> str:
    """
    Saves GNN state dict, hyperparameters, and preprocessing context safely.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    checkpoint = {
        'model_config': {
            'in_dim': encoder.in_dim,
            'hidden_dim': encoder.hidden_dim,
            'embedding_dim': encoder.embedding_dim,
            'dropout': encoder.dropout,
        },
        'encoder_state_dict': encoder.state_dict(),
        'discriminator_state_dict': discriminator.state_dict() if discriminator else None,
        'scaler_params': scaler_params,
        'threshold_info': threshold_info,
        'metadata': metadata,
    }

    torch.save(checkpoint, file_path)
    return file_path

def load_checkpoint(file_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Safely loads trained GNN checkpoint and initializes encoder on designated device.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{file_path}'.")

    # Use weights_only=False for complete dictionary loading with metadata
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)

    config = checkpoint['model_config']
    encoder = GraphSAGEEncoder(
        in_dim=config['in_dim'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        dropout=config['dropout']
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()

    return {
        'encoder': encoder,
        'model_config': config,
        'scaler_params': checkpoint.get('scaler_params', {}),
        'threshold_info': checkpoint.get('threshold_info', {}),
        'metadata': checkpoint.get('metadata', {}),
    }
