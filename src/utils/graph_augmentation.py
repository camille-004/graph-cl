"""Data augmentation functions."""
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def drop_random_edges(
    edge_index: torch.Tensor, p_drop: float = 0.2
) -> torch.Tensor:
    """Drop random edges from the input adjacency matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor where each column denotes an edge.
    p_drop : float
        Percentage of edges to drop

    Returns
    -------
    torch.Tensor
        Edge index tensor with dropped edges.
    """
    num_edges = edge_index.shape[1]
    num_edges_drop = int(p_drop * num_edges)
    mask = torch.ones(num_edges, dtype=bool)
    drop_indices = np.random.choice(num_edges, num_edges_drop, replace=False)
    mask[drop_indices] = False
    edge_index_dropped = edge_index[:, mask]

    return edge_index_dropped


def add_random_edges(
    edge_index: torch.Tensor, _num_nodes: int, p_add: float = 0.2
) -> torch.Tensor:
    """Add random edges from the input adjacency matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor where each column denotes an edge.
    _num_nodes : int
        Total number of edges.
    p_add : float
        Percentage of edges to drop

    Returns
    -------
    torch.Tensor
        Edge index tensor with added edges.
    """
    num_edges = edge_index.shape[1]
    num_edges_add = int(p_add * num_edges)
    edge_index_np = edge_index.cpu().numpy()
    edges_add = np.random.randint(
        low=0, high=_num_nodes, size=(2, num_edges_add)
    )

    # Ensure new edges are not duplicates
    for i in range(num_edges_add):
        while edges_add[:, i] in edge_index_np.T:
            edges_add[:, i] = np.random.randint(low=0, high=_num_nodes, size=2)

    # Add new edges to graph
    edge_index_added = torch.cat(
        [edge_index, torch.tensor(edges_add, dtype=torch.long)], dim=1
    )

    return edge_index_added


def edge_perturbation(_data: Data, p_drop: float, p_add: float) -> Data:
    """Edge perturbation.

    Perform edge perturbation on the input graph by dropping and adding
    random edges.

    Parameters
    ----------
    _data : Data
        Input graph.
    p_drop : float
        Percentage of edges to drop.
    p_add : float
        Percentage of edges to add.

    Returns
    -------
    Data
        Graph with perturbed edges.
    """
    start_time = time.time()

    edge_index = _data.edge_index
    _num_nodes = _data.num_nodes
    edge_index_dropped = drop_random_edges(edge_index, p_drop)
    edge_index_added = add_random_edges(edge_index_dropped, _num_nodes, p_add)

    _data.edge_index = edge_index_added

    end_time = time.time()
    print("Perturbation took {:.2f} seconds".format(end_time - start_time))

    return _data


def drop_random_nodes(_data: Data, p_drop: float = 0.2) -> Data:
    """Drop random nodes from the input graph.

    Parameters
    ----------
    _data : Data
        Input graph.
    p_drop : float
        Percentage of nodes to drop.

    Returns
    -------
    Data
        Graph with dropped nodes.
    """
    _num_nodes = _data.num_nodes
    num_nodes_drop = int(p_drop * _num_nodes)
    mask = torch.ones(_num_nodes, dtype=bool)
    drop_indices = torch.randperm(_num_nodes)[:num_nodes_drop]
    mask[drop_indices] = False

    # Drop from edge index
    edge_index = _data.edge_index
    rows, cols = edge_index
    mask_edge = mask[rows] & mask[cols]
    edge_index_dropped = edge_index[:, mask_edge]

    # Re-index edge index
    remaining_nodes = torch.arange(_num_nodes)[mask]
    map_new_indices = torch.full((_num_nodes,), -1, dtype=torch.long)
    map_new_indices[remaining_nodes] = torch.arange(remaining_nodes.size(0))
    edge_index_dropped = map_new_indices[edge_index_dropped]

    # Drop nodes from node features if present
    if _data.x is not None:
        _data.x = _data.x[mask]

    _data.edge_index = edge_index_dropped
    _data.num_nodes = remaining_nodes.size(0)

    return _data


def get_subgraph(
    _data: Data,
    p_sample: float = 0.2,
    walk_length: int = 10,
    max_attempts: int = 100,
) -> Data:
    """Extract a subgraph from the graph by performing a random walk.

    Parameters
    ----------
    _data : Data
        Input graph.
    p_sample : float
        Percentage of nodes to sample.
    walk_length : int
        Length of the random walk.
    max_attempts : int
        Maximum number of attempts to perform the random walk.

    Returns
    -------
    Data
        Subgraph data.
    """
    edge_index = _data.edge_index
    _num_nodes = _data.num_nodes
    num_nodes_sample = int(p_sample * _num_nodes)

    sampled_nodes = torch.tensor([], dtype=torch.long)
    attempts = 0

    while sampled_nodes.size(0) < num_nodes_sample and attempts < max_attempts:
        start_node = torch.randint(0, _num_nodes, (1,))
        walk_nodes = [start_node.item()]
        current_node = start_node.item()

        for _ in range(walk_length):
            neighbors = edge_index[1, edge_index[0] == current_node]
            if neighbors.size(0) == 0:
                break  # current_node is a leaf node
            current_node = neighbors[
                torch.randint(0, neighbors.size(0), (1,))
            ].item()
            walk_nodes.append(current_node)

        sampled_nodes = torch.unique(
            torch.cat(
                [sampled_nodes, torch.tensor(walk_nodes, dtype=torch.long)]
            )
        )
        attempts += 1

    subgraph_nodes, subgraph_edges = subgraph(sampled_nodes, edge_index)
    data_subgraph = _data.clone()
    data_subgraph.edge_index = subgraph_edges

    # If node features exist, extract subgraph node features
    if _data.x is not None:
        data_subgraph.x = _data.x[sampled_nodes]

    data_subgraph.num_nodes = sampled_nodes.size(0)

    return data_subgraph


def attribute_masking(_data: Data, p_mask: float = 0.2) -> Data:
    """Perform attribute masking on the input graph.

    Parameters
    ----------
    _data : Data
        Input graph.
    p_mask : float
        Percentage of node attributes to mask.

    Returns
    -------
    Data
        Graph with masked attributes.
    """
    _num_nodes = _data.num_nodes
    num_node_features = _data.x.size(1)
    num_features_mask = int(p_mask * num_node_features)
    mask = torch.ones(num_node_features, dtype=bool)
    mask_indices = torch.randperm(num_node_features)[:num_features_mask]
    mask[mask_indices] = False
    _data.x_masked = _data.x.clone()
    _data.x_masked[:, mask] = 0
    _data.attr_mask = mask

    return _data
