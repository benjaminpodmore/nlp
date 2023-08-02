import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


def collate_fn(batch, model=AutoModel.from_pretrained("bert-base-uncased")):
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    attention_masks = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
    outputs = model(input_ids=input_ids, attention_mask=attention_masks)

    return outputs.last_hidden_state, torch.tensor([x["label"] for x in batch])


def get_dataloader_and_vocab(batch_size, split="train"):
    dataset = load_dataset("rotten_tomatoes", split=split)
    # tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    encoded_dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
                                  batched=True,
                                  batch_size=None)
    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader, vocab


if __name__ == "__main__":
    train_dataloader, vocab = get_dataloader_and_vocab(batch_size=10)
    batch_input, batch_output = next(iter(train_dataloader))
