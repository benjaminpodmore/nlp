import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from functools import partial


def tokenize(src_tokenizer, src_key, tgt_tokenizer, tgt_key, batch):
    src_out = src_tokenizer([x[src_key] for x in batch], padding=True, truncation=True)
    tgt_out = tgt_tokenizer([x[tgt_key] for x in batch], padding=True, truncation=True)

    output = {"src_input_ids": src_out["input_ids"], "tgt_input_ids": tgt_out["input_ids"],
              "src_attention_masks": src_out["attention_mask"], "tgt_attention_masks": tgt_out["attention_mask"]}
    return output


def collate_fn(model, batch):
    src_input_ids = torch.stack([torch.tensor(x["src_input_ids"]) for x in batch])
    # src_attention_masks = torch.stack([torch.tensor(x["src_attention_masks"]) for x in batch])
    tgt_input_ids = torch.stack([torch.tensor(x["tgt_input_ids"]) for x in batch])
    # tgt_attention_masks = torch.stack([torch.tensor(x["tgt_attention_masks"]) for x in batch])
    #
    # src_outputs = model(src_input_ids, attention_mask=src_attention_masks)
    # tgt_outputs = model(tgt_input_ids, attention_mask=tgt_attention_masks)

    return src_input_ids, tgt_input_ids
    # return en_outputs.last_hidden_state, nl_outputs.last_hidden_state


def get_dataloader_and_vocab(batch_size, split):
    en_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    it_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

    dataset = load_dataset("opus100", "en-it", split=split)
    dataset_sample = Dataset.from_dict(dataset[:10000])

    encoded_dataset = dataset_sample.map(lambda batch: tokenize(en_tokenizer, "en", it_tokenizer, "it",
                                                                batch["translation"]), batched=True, batch_size=None)

    # TODO implement random_split

    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, model))
    vocab = en_tokenizer.get_vocab()

    return dataloader, vocab


if __name__ == "__main__":
    train_dataloader, vocab = get_dataloader_and_vocab(64, "train")
    batch = next(iter(train_dataloader))
    print(len(vocab))
