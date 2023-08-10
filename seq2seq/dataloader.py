import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer
from functools import partial


def tokenize(tokenizer, batch):
    src_out = tokenizer(batch["en"], padding=True, truncation=True)
    tgt_out = tokenizer(batch["de"], padding=True, truncation=True)

    return {
        "src_input_ids": src_out["input_ids"],
        "src_attention_mask": src_out["attention_mask"],
        "tgt_input_ids": tgt_out["input_ids"],
        "tgt_attention_mask": tgt_out["attention_mask"]
    }


def collate_fn(batch):
    inputs = torch.stack([torch.tensor([x["src_input_ids"] for x in batch])])
    labels = torch.stack([torch.tensor([x["tgt_input_ids"] for x in batch])])
    return inputs, labels


def get_dataloader_and_vocab(batch_size):
    dataset = load_dataset("bentrevett/multi30k")
    train_dataset = Dataset.from_dict(dataset["train"][0:28000])
    validation_dataset = Dataset.from_dict(dataset["train"][28000:])
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab = tokenizer.get_vocab()

    train_dataset_tokenized = train_dataset.map(lambda batch: tokenize(tokenizer, batch), batched=True, batch_size=None)
    validation_dataset_tokenized = validation_dataset.map(lambda batch: tokenize(tokenizer, batch), batched=True, batch_size=None)
    test_dataset_tokenized = test_dataset.map(lambda batch: tokenize(tokenizer, batch), batched=True, batch_size=None)

    train_dataloader = DataLoader(train_dataset_tokenized, batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset_tokenized, batch_size, shuffle=True,
                                       collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset_tokenized, batch_size, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, validation_dataloader, test_dataloader, vocab


if __name__ == "__main__":
    train_dataloader, validation_dataloader, test_dataloader, vocab = get_dataloader_and_vocab(64)
    batch = next(iter(train_dataloader))
    print(len(vocab))
