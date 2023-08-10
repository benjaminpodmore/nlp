import random
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, device):
        super().__init__()

        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size, output_size)

        self.vocab_size = vocab_size
        self.device = device

    def forward(self, x, labels, teacher_p):
        batch_size = x.shape[1]
        seq_length = labels.shape[0]

        encoder_states, hidden, cell = self.encoder(x)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        outputs = torch.zeros((seq_length, batch_size, self.vocab_size)).to(self.device)

        prev_token = labels[0].unsqueeze(0)
        for t in range(1, seq_length):
            preds, hidden, cell = self.decoder(prev_token, encoder_states, hidden, cell)
            outputs[t] = preds

            prev_token = labels[t].unsqueeze(0) if random.random() < teacher_p else preds.argmax(2)

        return outputs
