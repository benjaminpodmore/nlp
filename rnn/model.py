import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)
        combined = self.dropout(combined)
        hidden = F.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)

        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(64, self.hidden_size))
