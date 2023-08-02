import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def collate_fn(batch):
    batch_input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    batch_attention_masks = torch.stack([torch.tensor(x["input_ids"]) for x in batch])

    model = AutoModel.from_pretrained("distilbert-base-uncased")
    outputs = model(batch_input_ids, attention_mask=batch_attention_masks)

    batch_inputs = outputs.last_hidden_state
    batch_outputs = torch.stack([torch.tensor(x["label"]) for x in batch])
    return batch_inputs, batch_outputs


def get_dataloader_and_vocab(batch_size, split):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("rotten_tomatoes", split=split)
    encoded_dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding=True, truncation=True), batched=True, batch_size=None)

    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vocab = tokenizer.get_vocab()

    return dataloader, vocab


if __name__ == "__main__":
    train_dataloader, vocab = get_dataloader_and_vocab(64, "train")
    batch_inputs, batch_outputs = next(iter(train_dataloader))
    print(batch_inputs.shape, batch_outputs.shape)
