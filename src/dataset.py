from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

dataset = datasets.MNIST
dataset_name = dataset.__name__.lower()
input_size = 28 * 28

transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5,), (0.5,))
                                ])
# Train set. Here we sort the MNIST by digits and disable data shuffling
train_dataset = dataset(root='./data', train=True, download=True, transform=transform)
sorted_indices = sorted(range(len(train_dataset) // 1), key=lambda idx: train_dataset.targets[idx])
train_subset = Subset(train_dataset, sorted_indices)
train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)

# MultiTask training sets
train_loader_tasks = []
indices = []
for k in range(5):
    indices.append(list(
        filter(lambda idx: train_dataset.targets[idx] in range(k * 2, k * 2 + 2), range(len(train_dataset)))))
    train_loader_tasks.append(
        DataLoader(Subset(train_dataset, indices[-1]), batch_size=64, shuffle=True))

# Test set
test_dataset = dataset(root='./data', train=False, download=True, transform=transform)
test_subset = Subset(test_dataset, range(len(test_dataset) // 1))
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)