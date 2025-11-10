import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from typing import List, Optional

model_name = "google/flan-t5-small"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class Transformer(nn.Module):
    pass


class MyDataset(Dataset):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        tokenizer: PreTrainedTokenizer,
        max_samples: Optional[int] = None,
    ):
        self.input_ids = []
        self.output_ids = []

        for i in range(0, len(inputs), 64):
            input_ids = tokenizer(inputs[i : i + 64], return_tensors="pt")
            self.input_ids.extend(input_ids)
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        pass
