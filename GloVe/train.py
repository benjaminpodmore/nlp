from dataloader import get_dataloader_and_vocab


def train(batch_size):
    train_dataloader, vocab = get_dataloader_and_vocab(batch_size, "train")
    print(1)


if __name__ == "__main__":
    batch_size = 10
    train(batch_size)
