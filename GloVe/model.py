import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, embedding_size, vocab_size, x_max=100, alpha=3/4):
        super().__init__()

        self.focal_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.focal_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

    def forward(self):
        pass
