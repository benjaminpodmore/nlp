import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)
        hidden = self.in2hidden(combined)
        output = self.hidden2output(hidden)
        output = F.log_softmax(output, dim=1)

        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(64, self.hidden_size))
