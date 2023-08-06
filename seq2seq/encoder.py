import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout_p):
        super().__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
