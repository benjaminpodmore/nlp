import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader_and_vocab
from transformers import AutoModel
from model import Seq2Seq
from trainer import Trainer


def train(batch_size,  embedding_dim, hidden_size, epochs, train_steps, val_steps, clip):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    train_dataloader, validation_dataloader, test_dataloader, vocab = get_dataloader_and_vocab(batch_size)
    vocab_size = len(vocab)

    model = Seq2Seq(vocab_size, embedding_dim, hidden_size, vocab_size, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, device, epochs, optimizer, criterion, train_dataloader, train_steps, validation_dataloader, val_steps, clip)
    trainer.train()


if __name__ == "__main__":
    BATCH_SIZE = 128
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    EPOCHS = 10
    TRAIN_STEPS = 1
    VALIDATION_STEPS = 1
    CLIP = 1
    train(BATCH_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, EPOCHS, TRAIN_STEPS, VALIDATION_STEPS, CLIP)
