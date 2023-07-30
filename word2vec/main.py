import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

CBOW_N_WORDS = 4


class CBOW(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300, max_norm=1)
        self.linear = nn.Linear(in_features=300, out_features=vocab_size)

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class Trainer:
    def __init__(self, model, epochs, optimizer, lr_scheduler, criterion, train_dataloader, train_steps, device):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.device = device

        self.loss = {"train": []}
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self.lr_scheduler.step()

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        print(epoch_loss)
        self.loss["train"].append(epoch_loss)


def collate_cbow(batch):
    batch_input, batch_output = [], []
    for text in batch:
        text_token_ids = text["input_ids"]
        if len(text_token_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        for idx in range(len(text_token_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_token_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)

    return batch_input, batch_output


def train(total_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    encoded_dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
                                  batched=True,
                                  batch_size=None)
    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=10, shuffle=True, collate_fn=collate_cbow)
    # train_dataset = collate_cbow(encoded_dataset["train"])

    model = CBOW(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, 10, optimizer, lr_scheduler, criterion, train_dataloader=train_dataloader, train_steps=10,
                      device=device)
    trainer.train()


if __name__ == "__main__":
    train(total_epochs=100, lr=0.01)
