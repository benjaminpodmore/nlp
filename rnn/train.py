import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader_and_vocab
from model import RNN
from trainer import Trainer


def train(epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    train_dataloader, vocab = get_dataloader_and_vocab(batch_size=batch_size, split="train")
    test_dataloader, _ = get_dataloader_and_vocab(batch_size=batch_size, split="validation")
    model = RNN(input_size=78*768, hidden_size=78*768, output_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, epochs, optimizer, criterion, train_dataloader, val_dataloader=test_dataloader, device=device)
    trainer.train()


if __name__ == "__main__":
    train(epochs=50, batch_size=1)
