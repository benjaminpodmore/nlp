import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from constants import CBOW_N_WORDS, MAX_SEQUENCE_LENGTH


def collate_cbow(batch):
    batch_input, batch_output = [], []
    for text in batch:
        text_token_ids = text["input_ids"]
        if len(text_token_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_token_ids = text_token_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_token_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_token_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)

    return batch_input, batch_output


def get_dataloader_and_vocab(batch_size, split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    # tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    encoded_dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
                                  batched=True,
                                  batch_size=None)
    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_cbow)

    return dataloader, vocab
