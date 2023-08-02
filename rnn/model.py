import torch
import torch.nn as nn
import torch.nn.functional as F

from sympy.abc import x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        output = self.fc(hidden[-1])
        output = F.log_softmax(output, dim=1)
        return output
