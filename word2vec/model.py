import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, max_norm=1)
        self.linear = nn.Linear(in_features=embedding_size, out_features=vocab_size)

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x
