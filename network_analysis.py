from _reload_utils import reload_model, infer_hidden_layers
from _network_utils import build_full_mlp_graph, compute_centralities
from viz import summarize_centralities, visualize_all_centralities

def analyze_model(model_path):
    """
    Reloads a model, builds a full directed graph of the MLP,
    computes centralities, and summarizes the results.
    """
    model = reload_model(model_path, device='cpu')
    model.eval()

    print(f"Inferred hidden layers: {infer_hidden_layers(model)}")

    # --- Build and analyze full model graph ---
    full_graph = build_full_mlp_graph(model)
    centralities = compute_centralities(full_graph)

    # --- Package and summarize ---
    layer_results = [{
        'layer': 'full_model',
        'graph': full_graph,
        'centrality': centralities
    }]

    summarize_centralities(layer_results, top_k=5)
    visualize_all_centralities(layer_results)
    return layer_results

if __name__ == '__main__':
    model_path = './mlp/mlp_mnist357.pt'
    analyze_model(model_path)
