from config import Config
from data import load_urbandict, ModelDataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    config = Config()

    sources, targets = load_urbandict()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = ModelDataset(sources, targets, tokenizer, config.batch_size)