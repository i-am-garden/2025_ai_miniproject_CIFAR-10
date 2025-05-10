# src/datasets.py
from torchvision import datasets, transforms

def get_baseline_loaders(batch_size=128):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test  = transforms.ToTensor()

    train_set = datasets.CIFAR10(root='../data', train=True,
                                 download=True, transform=tf_train)
    test_set  = datasets.CIFAR10(root='../data', train=False,
                                 download=True, transform=tf_test)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
