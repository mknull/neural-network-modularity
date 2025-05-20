import matplotlib.pyplot as plt
import json
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_history(history, save_dir, filename_prefix="training_history"):
    """
    Plot training/validation/test loss and accuracy curves.
    Save individual and combined figures as PNG and PDF.

    Args:
        history (dict): Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                        and optionally 'test_loss', 'test_acc'.
        save_dir (str): Directory to save the plot images.
        filename_prefix (str): Prefix for saved image filenames.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Styling for publication-quality plots
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "serif",
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    # Colors: colorblind-friendly palette
    train_color = "#1f77b4"  # blue
    val_color = "#ff7f0e"    # orange
    test_color = "#2ca02c"   # green

    # ----- Individual Loss Plot -----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, history['train_loss'], label='Train Loss', color=train_color, marker='o')
    ax.plot(epochs, history['val_loss'], label='Validation Loss', color=val_color, marker='s')
    if 'test_loss' in history:
        ax.axhline(history['test_loss'], color=test_color, linestyle='--', label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()
    ax.minorticks_on()
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, f"{filename_prefix}_loss.png"))
    fig.savefig(os.path.join(save_dir, f"{filename_prefix}_loss.pdf"))
    plt.close(fig)

    # ----- Individual Accuracy Plot -----
    if history['train_acc'][0] is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, history['train_acc'], label='Train Accuracy', color=train_color, marker='o')
        ax.plot(epochs, history['val_acc'], label='Validation Accuracy', color=val_color, marker='s')
        if 'test_acc' in history:
            ax.axhline(history['test_acc'], color=test_color, linestyle='--', label='Test Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()
        ax.minorticks_on()
        fig.tight_layout()

        fig.savefig(os.path.join(save_dir, f"{filename_prefix}_accuracy.png"))
        fig.savefig(os.path.join(save_dir, f"{filename_prefix}_accuracy.pdf"))
        plt.close(fig)

    # ----- Combined Mega Figure -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss subplot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color=train_color, marker='o')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color=val_color, marker='s')
    if 'test_loss' in history:
        ax1.axhline(history['test_loss'], color=test_color, linestyle='--', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend()
    ax1.minorticks_on()

    # Accuracy subplot
    if history['train_acc'][0] is not None:
        ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color=train_color, marker='o')
        ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', color=val_color, marker='s')
        if 'test_acc' in history:
            ax2.axhline(history['test_acc'], color=test_color, linestyle='--', label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.legend()
        ax2.minorticks_on()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{filename_prefix}_combined.png"))
    fig.savefig(os.path.join(save_dir, f"{filename_prefix}_combined.pdf"))
    plt.close(fig)
    with open(os.path.join(save_dir, f"{filename_prefix}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

import os
import matplotlib.pyplot as plt

def summarize_centralities(layer_results, top_k=5, save_dir="plots"):
    """
    Print summary and plot top-k nodes by each centrality metric per layer.

    Args:
        layer_results (list): List of dicts, each with keys 'layer', 'graph', 'centrality'.
        top_k (int): Number of top nodes to show/plot per metric.
        save_dir (str): Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Color palette (colorblind-friendly)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for res in layer_results:
        layer_name = res['layer']
        print(f"\nLayer: {layer_name}")

        for metric_idx, (metric, values) in enumerate(res['centrality'].items()):
            if not values:
                print(f"  {metric:<15}: No data")
                continue

            # Sort descending by centrality value
            sorted_vals = sorted(values.items(), key=lambda x: -x[1])
            top_nodes = sorted_vals[:top_k]

            # Print summary
            print(f"  {metric:<15}: {top_nodes}")

            # Prepare data for plotting
            nodes, scores = zip(*top_nodes)

            # Plot bar chart
            plt.figure(figsize=(8, 4))
            bars = plt.bar(nodes, scores, color=colors[metric_idx % len(colors)])
            plt.title(f"Top {top_k} nodes by {metric} ({layer_name})")
            plt.ylabel("Centrality Score")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save plot
            filename = f"{layer_name}_{metric}_top{top_k}.png".replace(" ", "_")
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
def visualize_graph_with_centrality(G, centrality_scores, metric_name, layer_name, cmap='viridis', save_dir='plots/centralities'):
    """
    Visualizes the graph using a spring layout with nodes colored by a specific centrality metric.
    Saves the plot to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)

    values = [centrality_scores.get(node, 0.0) for node in G.nodes()]
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    node_colors = cm.get_cmap(cmap)(norm(values))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'{metric_name} centrality')

    ax.set_title(f"{layer_name} - {metric_name}")
    ax.set_axis_off()
    fig.tight_layout()

    filename = f"{layer_name}_{metric_name}.png".replace(" ", "_").lower()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {path}")

def visualize_all_centralities(layer_results, metrics=('in_degree', 'out_degree', 'betweenness', 'information')):
    """
    Loop through layers and metrics, generating one plot per centrality.
    """
    for layer_info in layer_results:
        layer_name = layer_info['layer']
        G = layer_info['graph']
        centrality = layer_info['centrality']

        for metric in metrics:
            if metric in centrality:
                visualize_graph_with_centrality(
                    G,
                    centrality_scores=centrality[metric],
                    metric_name=metric,
                    layer_name=layer_name
                )