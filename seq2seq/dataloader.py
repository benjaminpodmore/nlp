import torch
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from functools import partial


def tokenize(tokenizer, batch):
    en_out = tokenizer([x["en"] for x in batch], padding=True, truncation=True)
    nl_out = tokenizer([x["it"] for x in batch], padding=True, truncation=True)

    output = {"en_input_ids": en_out["input_ids"], "nl_input_ids": nl_out["input_ids"],
              "en_attention_masks": en_out["attention_mask"], "nl_attention_masks": nl_out["attention_mask"]}
    return output


def collate_fn(model, batch):
    en_input_ids = torch.stack([torch.tensor(x["en_input_ids"]) for x in batch])
    en_attention_masks = torch.stack([torch.tensor(x["en_attention_masks"]) for x in batch])
    nl_input_ids = torch.stack([torch.tensor(x["nl_input_ids"]) for x in batch])
    nl_attention_masks = torch.stack([torch.tensor(x["nl_attention_masks"]) for x in batch])

    en_outputs = model(en_input_ids, attention_mask=en_attention_masks)
    nl_outputs = model(nl_input_ids, attention_mask=nl_attention_masks)

    return en_input_ids, nl_input_ids
    # return en_outputs.last_hidden_state, nl_outputs.last_hidden_state


def get_dataloader_and_vocab(batch_size, split):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

    dataset = load_dataset("opus100", "en-it", split=split)

    encoded_dataset = dataset.map(lambda batch: tokenize(tokenizer, batch["translation"]), batched=True, batch_size=None)

    # TODO implement random_split

    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, model))
    vocab = tokenizer.get_vocab()

    return dataloader, vocab


if __name__ == "__main__":
    train_dataloader, vocab = get_dataloader_and_vocab(64, "train")
    batch = next(iter(train_dataloader))
    print(len(vocab))
