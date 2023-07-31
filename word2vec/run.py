import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from dataloader import get_dataloader_and_vocab
from model import CBOW
from trainer import Trainer
from constants import WORD_EMBEDDING_DIM


def train(batch_size, total_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    train_dataloader, vocab = get_dataloader_and_vocab(batch_size, "train")
    val_dataloader, _ = get_dataloader_and_vocab(batch_size, "validation")

    model = CBOW(len(vocab), WORD_EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model,
                      total_epochs,
                      optimizer,
                      lr_scheduler,
                      criterion,
                      train_dataloader=train_dataloader,
                      train_steps=20,
                      val_dataloader=val_dataloader,
                      val_steps=20,
                      device=device)
    trainer.train()


if __name__ == "__main__":
    train(batch_size=50, total_epochs=10, lr=0.01)
