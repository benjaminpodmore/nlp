import torch
import torch.optim as optim
from model import RNN
from trainer import Trainer
from dataloader import get_dataloader_and_vocab


def train(batch_size, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    train_dataloader, vocab = get_dataloader_and_vocab(batch_size, "train")
    val_dataloader, _ = get_dataloader_and_vocab(batch_size, "validation")

    model = RNN(768, 300, 2)
    lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=10)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(device, model, epochs, optimizer, scheduler, criterion, train_dataloader, 10, val_dataloader, 10)
    trainer.train()


if __name__ == "__main__":
    train(64, 20)
