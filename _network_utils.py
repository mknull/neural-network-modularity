import torch
import networkx as nx
from torch import nn
from _reload_utils import reload_model

def build_full_mlp_graph(model):
    """
    Construct a full graph of the MLP from input to output neurons across layers.
    Each node is named by layer and index: e.g., 'L0_3', 'L2_5'.
    """
    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]
    G = nx.DiGraph()

    prev_layer_nodes = [f"L0_{i}" for i in range(layers[0].in_features)]

    for layer_idx, layer in enumerate(layers):
        weight = layer.weight.detach().cpu().numpy()
        curr_layer_nodes = [f"L{layer_idx + 1}_{j}" for j in range(weight.shape[0])]

        for i, src in enumerate(prev_layer_nodes):
            for j, tgt in enumerate(curr_layer_nodes):
                G.add_edge(src, tgt, weight=weight[j, i])

        prev_layer_nodes = curr_layer_nodes

    return G


def compute_centralities(G):
    # 1) Precompute a cost attribute as inverse absolute weight
    for u, v, data in G.edges(data=True):
        data['cost'] = 1.0 / (abs(data['weight']) + 1e-8)

    # 2) Degree-centralities on abs(weight)
    in_deg = dict(G.in_degree(weight=lambda u, v, d: abs(d['weight'])))
    out_deg = dict(G.out_degree(weight=lambda u, v, d: abs(d['weight'])))

    # 3) Betweenness using cost as path length
    betw = nx.betweenness_centrality(G, weight='cost', normalized=True)

    centralities = {
        'in_degree': in_deg,
        'out_degree': out_deg,
        'betweenness': betw,
    }

    # 4) Try information centrality on undirected positive-cost graph
    try:
        U = G.to_undirected()
        for u, v, data in U.edges(data=True):
            data['weight'] = data.get('cost', 1.0)  # use cost as resistance
        info = nx.current_flow_closeness_centrality(U, weight='weight')
        centralities['information'] = info
    except Exception as e:
        print(f"[Info] Skipping information centrality: {e}")
        centralities['information'] = {node: 0.0 for node in G.nodes()}

    return centralities


def weights_to_digraph(weight_tensor):
    """
    Converts a weight matrix to a bipartite DiGraph (used for single-layer analysis).
    Nodes named 'in_i' and 'out_j'.
    """
    W = weight_tensor.detach().cpu().numpy()
    num_input, num_output = W.shape
    G = nx.DiGraph()
    for i in range(num_input):
        for j in range(num_output):
            G.add_edge(f"in_{i}", f"out_{j}", weight=W[i, j])
    return G


