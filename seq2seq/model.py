import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_dim, device):
        super().__init__()

        self.output_dim = output_dim

        self.encoder = Encoder(input_size, embedding_dim, hidden_size, 1, 0.1)
        self.decoder = Decoder(input_size, embedding_dim, hidden_size, 1, output_dim, 0.1)

        self.device = device

    def forward(self, x, label, teaching_p):
        batch_size = x.shape[0]
        tgt_seq_len = label.shape[1]
        hidden, cell = self.encoder(x)

        preds = torch.zeros(batch_size, tgt_seq_len, self.output_dim).to(self.device)
        prev_token = label[:, 0].unsqueeze(0)
        for t in range(1, tgt_seq_len):
            pred, hidden, cell = self.decoder(prev_token, hidden, cell)
            preds[:, t] = pred
            prev_token = label[:, t].unsqueeze(0) if random.random() < teaching_p else pred.argmax(2)

        return preds
