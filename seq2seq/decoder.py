import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, output_size, dropout_p):
        super().__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden,  cell))

        pred = self.fc(outputs)
        return pred, hidden, cell
