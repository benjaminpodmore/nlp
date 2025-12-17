from dataclasses import dataclass
import torch

@dataclass
class Config:
    model_name = "google/flan-t5-small"

    batch_size: int = 64


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")