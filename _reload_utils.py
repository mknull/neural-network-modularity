import torch
import models

def save_model(model, model_class_name, model_args, path):
    checkpoint = {
        "model_class": model_class_name,
        "model_args": model_args,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def reload_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not all(k in checkpoint for k in ('model_class', 'model_args', 'state_dict')):
        raise KeyError("Checkpoint missing keys: 'model_class', 'model_args', or 'state_dict'")

    model_class_name = checkpoint['model_class']
    model_args = checkpoint['model_args']
    state_dict = checkpoint['state_dict']

    try:
        model_class = getattr(models, model_class_name)
    except AttributeError:
        raise ValueError(f"Model class '{model_class_name}' not found in models.py")

    model = model_class(**model_args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_hidden_layers(model):
    """
    Given the MLPClassifier instance, infer hidden layer sizes
    by inspecting nn.Linear layers inside model.model Sequential,
    skipping the first Flatten and the last output Linear.
    """
    layers = list(model.model.children())

    # Filter only Linear layers (skip Flatten and ReLU)
    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]

    # Hidden layers: all except last Linear (output)
    hidden_linears = linear_layers[:-1]

    hidden_dims = [layer.out_features for layer in hidden_linears]
    return hidden_dims
