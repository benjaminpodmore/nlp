import polars as pl
from torch.utils.data import Dataset, DataLoader

def load_urbandict(path="urbandict-word-defs.csv", max_samples: int | None = None):
    df = pl.read_csv(path, has_header=True, truncate_ragged_lines=True, infer_schema_length=0)

    df_clean = df.filter(
        pl.col("definition").is_not_null() & 
        pl.col("word").is_not_null()
    )

    df_clean = df_clean.with_columns([
        pl.col("up_votes").cast(pl.Int64, strict=False),
        pl.col("down_votes").cast(pl.Int64, strict=False)
    ])

    df_clean = df_clean.drop_nulls(subset=["up_votes", "down_votes"])

    if max_samples:
        df_clean = df_clean[:max_samples]

    definitions = df_clean["definition"].to_list()
    words = df_clean["word"].to_list()

    return definitions, words

class ModelDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, batch_size=1):
        self.input_ids = []
        self.input_attention_masks = []
        self.target_ids = []
        self.target_attention_masks = []

        for i in range(0, len(inputs), batch_size):
            batch_input = tokenizer(inputs[i:i + batch_size], return_tensors="pt", truncation=True, padding=True)
            batch_target = tokenizer(outputs[i:i + batch_size], return_tensors="pt", truncation=True, padding=True)

            self.input_ids.extend(batch_input["input_ids"])
            self.input_attention_masks.extend(batch_input["attention_mask"])
            self.target_ids.extend(batch_target["input_ids"])
            self.target_attention_masks.extend(batch_target["attention_mask"])


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
                    "input_ids": self.input_ids[idx],
                    "input_attention_masks": self.input_attention_masks[idx],
                    "target_ids": self.target_ids[idx],
                    "target_attention_masks": self.target_attention_masks[idx]
                }