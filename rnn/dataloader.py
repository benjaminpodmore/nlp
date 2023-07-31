import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def collage_multiclass(batch):
    batch_input, batch_output = [], []

    for input, label in batch:
        output = torch.zeros(10)
        output[label] = 1
        batch_input.append(input)
        batch_output.append(output)


    batch_input = torch.stack(batch_input)
    batch_output = torch.stack(batch_output)

    return batch_input, batch_output


def get_dataloader(batch_size=64,is_train=True):
    dataset = MNIST(root="data", train=is_train, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collage_multiclass)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataloader
