from torch.utils.data import Dataset

class FilteredMNIST(Dataset):
    """
    Wraps an MNIST dataset but only keeps samples with labels in {3, 5, 7}.
    Re-labels the targets to 0, 1, 2 respectively.
    """
    def __init__(self, mnist_dataset):
        self.data = []
        self.targets = []
        self.label_map = {3: 0, 5: 1, 7: 2}

        for img, label in mnist_dataset:
            if label in self.label_map:
                self.data.append(img)
                self.targets.append(self.label_map[label])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
