from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def collate_fn(batch):
    return batch


def get_dataloader_and_vocab(batch_size, split):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    encoded_dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
                                  batched=True,
                                  batch_size=None)
    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return dataloader, vocab
