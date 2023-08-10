import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, encoder_states, hidden):
        encoder_states = self.fc(encoder_states)
        encoder_states = encoder_states.permute(1, 0, 2)
        hidden = hidden.permute(1, 2, 0)

        attention_scores = torch.bmm(encoder_states, hidden)
        attention_weights = F.softmax(attention_scores, dim=1)

        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), encoder_states)

        return context_vector, attention_weights
