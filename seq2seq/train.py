import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader_and_vocab
from model import Seq2Seq
from trainer import Trainer


def train(batch_size, epochs, train_steps, val_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device {device}.")

    train_dataloader, vocab = get_dataloader_and_vocab(batch_size, "train")
    val_dataloader, _ = get_dataloader_and_vocab(batch_size, "validation")

    model = Seq2Seq(len(vocab), 300, 128, len(vocab), device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=40)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(device, model, epochs, optimizer, lr_scheduler, criterion, train_dataloader, train_steps, val_dataloader, val_steps)

    trainer.train()


if __name__ == "__main__":
    train(10, 100, 5, 5)