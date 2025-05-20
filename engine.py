import torch
import torch.nn as nn


def model_type(model):
    """
    Determine model type: 'vae', 'generator', or 'classifier'
    """
    clsname = model.__class__.__name__.lower()

    # Prefer explicit model.task or model.model_type if available
    if hasattr(model, 'model_type'):
        return model.model_type.lower()
    if hasattr(model, 'task'):
        return model.task.lower()

    # Fallback: class name heuristics
    if 'vae' in clsname:
        return 'vae'
    elif 'transformer' in clsname or 'vit' in clsname:
        return 'generator'  # or 'classifier' if ViT is for classification
    elif hasattr(model, 'loss_function'):
        return 'vae'
    else:
        return 'classifier'


def step(model, batch, criterion, device, is_train=True, optimizer=None):
    x, y = batch
    x, y = x.to(device), y.to(device)

    if is_train:
        optimizer.zero_grad()

    mtype = model_type(model)

    if mtype == 'vae':
        recon, mu, logvar = model(x)
        loss = model.loss_function(recon, x, mu, logvar)
        output = mu
        acc = None

    elif mtype == 'generator':
        output = model(x)
        if hasattr(model, 'loss_function'):
            loss = model.loss_function(output, x)
        else:
            loss = criterion(output, x)
        acc = None

    else:  # classifier
        output = model(x)
        loss = criterion(output, y)
        acc = (output.argmax(1) == y).float().mean().item()

    if is_train:
        loss.backward()
        optimizer.step()

    return loss.item(), acc


def run_epoch(model, dataloader, criterion, optimizer, device, mode='train'):
    is_train = mode == 'train'
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0
    has_acc = False

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            loss, acc = step(model, batch, criterion, device, is_train, optimizer)
            bs = len(batch[0])
            total_loss += loss * bs
            count += bs
            if acc is not None:
                total_acc += acc * bs
                has_acc = True

    avg_loss = total_loss / count
    avg_acc = total_acc / count if has_acc else None
    return avg_loss, avg_acc


def train_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, callback=None):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, mode='train')
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, mode='val')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:2d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}", end='')
        if train_acc is not None:
            print(f", Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        else:
            print()

        if callback:
            callback(epoch, model, history)

    return history


def test(model, test_loader, criterion, device):
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, mode='test')
    print(f"Test Loss = {test_loss:.4f}", end='')
    if test_acc is not None:
        print(f", Test Accuracy = {test_acc:.4f}")
    else:
        print()
    return test_loss, test_acc


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