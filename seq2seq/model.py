import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = Encoder(30522, 300, 128, 1, 0.1)
        self.decoder = Decoder(30522, 300, 128, 1, 30522, 0.1)

    def forward(self, x, label, teaching_p):
        hidden, cell = self.encoder(x)

        preds = torch.zeros(10, 285, 30522).to(torch.device("cuda"))
        prev_token = label[:, 0].unsqueeze(0)
        for t in range(1, 285):
            pred, hidden, cell = self.decoder(prev_token, hidden, cell)
            preds[:, t] = pred
            prev_token = label[:, t].unsqueeze(0) if random.random() < teaching_p else pred.argmax(2)

        return preds
