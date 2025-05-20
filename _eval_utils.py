import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from filtered_MNIST import FilteredMNIST
from tqdm import tqdm

def evaluate_per_class(model, batch_size=256):
    """
    Evaluate classification accuracy on digits 3, 5, 7 separately using FilteredMNIST.
    Assumes model input is flattened images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.ToTensor()
    test_raw = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_dataset = FilteredMNIST(test_raw)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating per class"):
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)  # flatten

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            for cls in [0, 1, 2]:
                mask = (y == cls)
                correct[cls] += (preds[mask] == cls).sum().item()
                total[cls] += mask.sum().item()

    label_map = {0: "digit 3", 1: "digit 5", 2: "digit 7"}
    print("\n--- Accuracy per class ---")
    for cls in [0, 1, 2]:
        acc = 100.0 * correct[cls] / total[cls] if total[cls] > 0 else 0.0
        print(f"{label_map[cls]}: {acc:.2f}%")

    return {cls: 100.0 * correct[cls] / total[cls] for cls in [0, 1, 2]}