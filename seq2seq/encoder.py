import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.fc_hidden = nn.Linear(2 * hidden_size, hidden_size)
        self.fc_cell = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x):
        embedding = self.embedding(x)
        encoder_states, (hidden, cell) = self.rnn(embedding)
        hidden = F.relu(self.fc_hidden(torch.cat((hidden[0], hidden[1]), dim=1)))
        cell = F.relu(self.fc_cell(torch.cat((cell[0], cell[1]), dim=1)))

        return encoder_states, hidden, cell
