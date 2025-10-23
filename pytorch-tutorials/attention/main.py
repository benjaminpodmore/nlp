import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init_(self, vocab_size: int, embed_dim: int, d_q: int):
        W_q = nn.Linear(d_q, d_q)
        embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        pass

    # Q = Wq * Embed(curr_word)
    # Q . K
    # replace forthcoming values (from "keys" we haven't seen yet) as -inf - this is called masking
    # softmax / root(d_k) - for numerical stability
    # that becomes the weight


if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
