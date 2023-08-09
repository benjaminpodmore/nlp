import torch
import torch.nn as nn
from attention import DotProductAttention


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.attention = DotProductAttention(hidden_size)
        self.rnn = nn.LSTM(input_size + hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        context_vector, attention_weights = self.attention(hidden, encoder_states)
        context_vector = context_vector.permute(1, 0, 2)
        outputs, (hidden, cell) = self.rnn(torch.cat((x, context_vector), dim=2), (hidden,  cell))

        pred = self.fc(outputs)
        return pred, hidden, cell
