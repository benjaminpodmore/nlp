from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def tokenize(tokenizer, batch):
    batch_inputs = tokenizer([x["en"] for x in batch], padding=True, truncation=True)
    batch_outputs = tokenizer([x["nl"] for x in batch], padding=True, truncation=True)
    return {"en": {**batch_inputs}, "nl": {**batch_outputs}}


def collate_fn(model, batch):
    en_input_ids = torch.stack([torch.tensor(x["en"]["input_ids"]) for x in batch])
    en_attention_masks = torch.stack([torch.tensor(x["en"]["attention_masks"]) for x in batch])
    nl_input_ids = torch.stack([torch.tensor(x["nl"]["input_ids"]) for x in batch])
    nl_attention_masks = torch.stack([torch.tensor(x["nl"]["attention_masks"]) for x in batch])

    en_outputs = model(en_input_ids, attention_mask=en_attention_masks)
    nl_outputs = model(nl_input_ids, attention_mask=nl_attention_masks)

    return en_outputs.last_hidden_state, nl_outputs.last_hidden_state


def get_dataloader_and_vocab(batch_size, split):
    dataset = load_dataset("ted_talks_iwslt", "nl_en_2016", split=split)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    encoded_dataset = dataset.map(lambda batch: tokenize(tokenizer, batch["translation"]), batched=True, batch_size=None)

    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, model))
    vocab = tokenizer.get_vocab()

    return dataloader, vocab


if __name__ == "__main__":
    train_dataloader, vocab = get_dataloader_and_vocab(64, "train")
    batch = next(iter(train_dataloader))
    print(len(vocab))
