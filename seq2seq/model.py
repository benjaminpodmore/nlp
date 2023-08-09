import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, device, emb_model):
        super().__init__()

        self.output_dim = output_dim

        self.encoder = Encoder(input_size, hidden_size, 1)
        self.decoder = Decoder(input_size, hidden_size, 1, output_dim)

        self.device = device
        self.emb_model = emb_model

    def forward(self, x, label_ids, label_embs, teaching_p):
        batch_size = x.shape[0]
        tgt_seq_len = label_ids.shape[1]
        encoder_states, hidden, cell = self.encoder(x)

        preds = torch.zeros(batch_size, tgt_seq_len, self.output_dim).to(self.device)
        prev_token = label_embs[:, 0].unsqueeze(0)
        for t in range(1, tgt_seq_len):
            pred, hidden, cell = self.decoder(prev_token, encoder_states, hidden, cell)
            preds[:, t] = pred
            with torch.no_grad():
                prev_emb = self.emb_model.embeddings(pred.argmax(2))
            prev_token = label_embs[:, t].unsqueeze(0) if random.random() < teaching_p else prev_emb

        return preds
