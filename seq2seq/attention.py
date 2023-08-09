import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, hidden, encoder_states):
        reduced_encoder_states = self.fc(encoder_states)
        attention_scores = torch.bmm(reduced_encoder_states, hidden.permute(1, 2, 0))
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), reduced_encoder_states)

        return context_vector, attention_weights
