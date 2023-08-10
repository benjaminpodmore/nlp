import torch
import torch.nn as nn
from attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_size)
        self.rnn = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        embedding = self.embedding(x)
        context_vector, _ = self.attention(encoder_states, hidden)
        context_vector = context_vector.permute(1, 0, 2)
        outputs, (hidden, cell) = self.rnn(torch.cat((embedding, context_vector), dim=2), (hidden, cell))
        outputs = self.fc(outputs)
        return outputs, hidden, cell
