import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from copy import deepcopy

from filtered_MNIST import FilteredMNIST
from _reload_utils import reload_model
from _inference_utils import find_high_in_low_between_neurons
from _eval_utils import evaluate_per_class

# Here we check whether we can detect modules in the perfect case that they are completely encapsulated, such as in
def ablate_neuron_and_top_inputs(model, hidden_layer_idx, tgt_idx, top_n_inputs):
    """
    Ablate the candidate hidden neuron and its top_n_inputs strongest feeding neurons.
    This means: zero all outgoing weights of those neurons to downstream layers.

    Args:
        model: A PyTorch MLP model with model.model as a Sequential of Linear layers
        hidden_layer_idx: index of the hidden layer in model.model (excluding input layer)
        tgt_idx: index of the neuron in that layer
        top_n_inputs: number of strongest input neurons to also ablate (from previous layer)
    """
    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]
    hidden_layers = layers[:-1]  # exclude output layer

    if hidden_layer_idx >= len(hidden_layers):
        raise IndexError(f"Hidden layer index {hidden_layer_idx} out of range.")

    # Weight matrix for this layer (target neuron's incoming weights)
    W = hidden_layers[hidden_layer_idx].weight.data  # [out, in]

    if tgt_idx >= W.shape[0]:
        raise IndexError(f"Target neuron index {tgt_idx} is out of bounds for layer {hidden_layer_idx}.")

    # Identify top-N strongest input neurons (from previous layer)
    incoming_weights = W[tgt_idx, :]  # 1D tensor of shape [in]
    strongest_inputs = torch.topk(incoming_weights.abs(), top_n_inputs).indices

    # 1. Ablate the target neuron's output (zero column in next layer)
    if hidden_layer_idx + 1 < len(hidden_layers):
        W_next = hidden_layers[hidden_layer_idx + 1].weight.data  # [out, in]
        if tgt_idx < W_next.shape[1]:
            W_next[:, tgt_idx] = 0.0

    # 2. Ablate output of each of the top-N strongest input neurons (zero their output columns)
    if hidden_layer_idx - 1 >= 0:
        W_prev_next = hidden_layers[hidden_layer_idx].weight.data  # [out, in]
        for src_idx in strongest_inputs:
            if src_idx < W_prev_next.shape[1]:
                W_prev_next[:, src_idx] = 0.0
    else:
        # We're in the first hidden layer, so input comes from input layer
        # Just zero the input connections to all layers for these neurons
        for src_idx in strongest_inputs:
            for layer in hidden_layers:
                if src_idx < layer.weight.data.shape[1]:
                    layer.weight.data[:, src_idx] = 0.0

    # 3. Finally, zero all incoming connections to the target neuron itself
    W[tgt_idx, :] = 0.0

def run_ablation_experiment(
    model_path,
    data_root="~/data",
    batch_size=128,
    top_k_in=4,
    top_percent_betw=0.95,
    top_n_inputs=4,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    test_raw = MNIST(root=data_root, train=False, download=True, transform=transform)

    label_map = {3: 0, 5: 1, 7: 2}
    digit_dataloaders = {
        digit: DataLoader(
            FilteredMNIST([(img, lbl) for img, lbl in test_raw if lbl == digit]),
            batch_size=batch_size, shuffle=False
        )
        for digit in label_map
    }

    candidate_nodes_by_layer = find_high_in_low_between_neurons(
        model_path,
        top_k_in=top_k_in,
        top_percent_betw=top_percent_betw,
    )

    print("\n=== Individual Ablation Results ===")

    for layer_idx, candidates in candidate_nodes_by_layer:
        for tgt in sorted(candidates):
            try:
                tgt_idx = int(tgt.split('_')[1])
            except Exception as e:
                print(f"Skipping {tgt}: could not parse index: {e}")
                continue

            if layer_idx == 0:
                print(f"Skipping {tgt}: in input layer.")
                continue

            print(f"\nAblating neuron {tgt} in layer {layer_idx} with top {top_n_inputs} inputs.")

            ablated_model = reload_model(model_path, device=device).eval()
            try:
                ablate_neuron_and_top_inputs(ablated_model, layer_idx - 1, tgt_idx, top_n_inputs)
            except IndexError as e:
                print(f"Skipping {tgt} due to error: {e}")
                continue

            digitwise_accs = evaluate_per_class(ablated_model, batch_size=batch_size)

            for cls, acc in digitwise_accs.items():
                digit = [k for k, v in label_map.items() if v == cls][0]
                # print(f"Digit {digit} (class {cls}): Accuracy after ablation = {acc:.2f}%")

    print("\n=== Ablation Experiment Complete ===")

if __name__ == "__main__":
    run_ablation_experiment("mlp/mlp_mnist357.pt")
