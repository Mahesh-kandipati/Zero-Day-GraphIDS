"""
Graph Export Utility.
Serializes graph network structures and topological statistics into JSON
for front-end Cytoscape.js / Chart.js interactive exploration.
"""

import json
from typing import Dict, Any, List
import networkx as nx
import numpy as np

def export_graph_to_cytoscape_json(
    G: nx.Graph,
    node_labels: Dict[int, str] = None,
    max_nodes: int = 150
) -> Dict[str, Any]:
    """
    Converts a NetworkX graph into Cytoscape.js compatible elements list.
    Caps max nodes for high-performance browser rendering.
    """
    if G.number_of_nodes() > max_nodes:
        # Sample sub-graph with highest degree centrality
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    elements = []

    for node_id in G.nodes():
        node_data = {
            'id': str(node_id),
            'label': f"Flow #{node_id}" if node_labels is None else str(node_labels.get(node_id, f"Node {node_id}")),
            'degree': G.degree(node_id),
            'clustering': round(float(nx.clustering(G, node_id)), 4) if G.number_of_nodes() > 1 else 0.0
        }
        elements.append({'data': node_data, 'group': 'nodes'})

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        edge_data = {
            'id': f"e_{u}_{v}",
            'source': str(u),
            'target': str(v),
            'weight': round(float(weight), 3)
        }
        elements.append({'data': edge_data, 'group': 'edges'})

    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'density': round(float(nx.density(G)), 4),
        'avg_degree': round(float(np.mean([d for n, d in G.degree()])), 2) if G.number_of_nodes() > 0 else 0.0,
        'avg_clustering': round(float(nx.average_clustering(G)), 4) if G.number_of_nodes() > 1 else 0.0,
    }

    return {
        'elements': elements,
        'stats': stats
    }
