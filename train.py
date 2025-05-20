import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from filtered_MNIST import FilteredMNIST  # Your filtered dataset for digits 3,5,7
from models import MLPClassifier  # Your MLP model
from engine import train_validate, test  # Your engine functions
from viz import plot_history
from _reload_utils import save_model

def main():
    parser = argparse.ArgumentParser(description="Train MLP on filtered MNIST digits 3,5,7 using engine.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-layers", type=str, default="128,64",
                        help="Comma-separated hidden layer sizes, e.g. 128,64")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse hidden layers argument
    hidden_layers = [int(h) for h in args.hidden_layers.split(",") if h.strip()]

    # Prepare data
    transform = transforms.ToTensor()

    # Download full MNIST datasets
    raw_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    raw_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Filter digits 3,5,7 using your FilteredMNIST class
    train_dataset = FilteredMNIST(raw_train)
    test_dataset = FilteredMNIST(raw_test)

    # Create train/val split (e.g., 90% train, 10% val)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Instantiate model
    # Output dim is 3 because digits are 3,5,7 = 3 classes
    input_dim = 28 * 28
    output_dim = 3
    model_args = {
        "input_dim": input_dim,
        "hidden_layers": hidden_layers,
        "output_dim": output_dim,
    }

    model = MLPClassifier(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=3).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train and validate using your engine
    history = train_validate(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    # Test final model
    test_loss, test_acc = test(model, test_loader, criterion, device)

    # Add test metrics to history for logging/visualization
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc

    # Save model checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "mlp_mnist357.pt")
    plot_history(history, args.save_dir)
    save_model(model, model.__class__.__name__, model_args, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
