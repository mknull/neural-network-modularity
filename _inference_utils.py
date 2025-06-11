import torch
import networkx as nx
from torch import nn
from _reload_utils import reload_model
from _network_utils import weights_to_digraph


def find_high_in_low_between_neurons(model_path, top_k_in=5, top_percent_betw=0.5):
    model = reload_model(model_path, device='cpu')
    model.eval()

    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]

    all_in_deg = {}
    all_betw = {}
    node_to_layer = {}

    # Collect centrality metrics for hidden layers
    for layer_idx, layer in enumerate(layers[1:], start=1):  # Start from second hidden layer
        W = layer.weight.data
        G = weights_to_digraph(W)

        in_deg = dict(G.in_degree(weight='weight'))
        betw = nx.betweenness_centrality(G, weight='weight', normalized=True)

        # Update global mappings
        for n in G.nodes:
            all_in_deg[f"{layer_idx}_{n}"] = in_deg[n]
            all_betw[f"{layer_idx}_{n}"] = betw[n]
            node_to_layer[f"{layer_idx}_{n}"] = layer_idx

    # Determine betweenness cutoff globally
    num_nodes = len(all_betw)
    cutoff = int(num_nodes * top_percent_betw)
    low_betw_nodes = {
        n for n, _ in sorted(all_betw.items(), key=lambda x: x[1])[:cutoff]
    }

    # Rank all nodes by in-degree, take top_k that are also low betweenness
    sorted_in_deg = sorted(all_in_deg.items(), key=lambda x: -x[1])
    candidates = []
    for node, _ in sorted_in_deg:
        if node in low_betw_nodes:
            candidates.append(node)
        if len(candidates) >= top_k_in:
            break

    # Return grouped by layer index for downstream logic
    candidate_nodes_by_layer = {}
    for node in candidates:
        layer_idx = node_to_layer[node]
        node_short = node.split("_", 1)[1]  # remove layer prefix
        candidate_nodes_by_layer.setdefault(layer_idx, set()).add(node_short)

    # Convert to list of (layer_idx, set(...)) as before
    return list(candidate_nodes_by_layer.items())

