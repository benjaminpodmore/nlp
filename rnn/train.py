import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader
from model import RNN
from trainer import Trainer


def train(epochs=10, train_steps=100, val_steps=100, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    train_dataloader = get_dataloader(batch_size=batch_size, is_train=True)
    test_dataloader = get_dataloader(batch_size=batch_size, is_train=False)
    model = RNN(input_size=28*28, hidden_size=28*28, output_size=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, epochs, optimizer, criterion, train_dataloader, train_steps, val_dataloader=test_dataloader, val_steps=val_steps, device=device)
    trainer.train()


if __name__ == "__main__":
    train(epochs=50, train_steps=64, val_steps=64, batch_size=64)
