import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = Encoder(768, 20, 1)
        self.decoder = Decoder(768, 20, 1, 768)

    def forward(self, x, label):
        hidden, cell = self.encoder(x)
        h0 = hidden[-1]
        c0 = cell[-1]

        preds = torch.zeros(1, 285, 768).to(torch.device("cuda"))
        prev_token = label[0][0].unsqueeze(0)
        for t in range(1, 285):
            pred, h0, c0 = self.decoder(prev_token, h0, c0)
            preds[0][t] = pred
            prev_token = label[0][t].unsqueeze(0) if random.random() < 0.5 else pred

        return preds
