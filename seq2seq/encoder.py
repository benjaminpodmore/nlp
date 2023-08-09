import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(2 * hidden_size, hidden_size)
        self.fc_cell = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x):
        outputs, (hidden, cell) = self.rnn(x)
        hidden = F.relu(self.fc_hidden(torch.cat((hidden[0], hidden[1]), dim=1)))
        cell = F.relu(self.fc_cell(torch.cat((cell[0], cell[1]), dim=1)))

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        return outputs, hidden, cell
