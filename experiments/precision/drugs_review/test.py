import pandas as pd
from functools import reduce
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
import re

data = pd.read_csv(
    "/Users/victor/Documents/projects/coreset/data/drugs_review/drugsComTrain_raw.tsv",
    sep="\t",
    index_col=0,
)


class TxtDataset(Dataset):
    pass


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, dim, context_size):
        self.emb = nn.Embedding(vocab_size, dim)
        self.hidden1 = nn.Linear(context_size * dim, 128)
        self.hidden2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        emb = self.emb(inputs)
        x = self.hidden1(emb)
        x = self.hidden2(x)
        logprob = F.log_softmax(x, dim=1)
        return logprob


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent.split()


def prepair_ngrams(sent, context_size=2):
    sent = clean_sent(sent)
    sent_size = len(sent)
    return [
        (sent[i : i + context_size], sent[i + context_size])
        for i in range(sent_size - context_size)
    ]


def build_vocab(dataset):
    vocab = " ".join(dataset)
    vocab = clean_sent(vocab)
    vocab = set(vocab)
    return {w: i for w, i in enumerate(vocab)}


if __name__ == "__main__":
    import torch

    review = data["review"].values

    vocab = build_vocab(review)
    ngrams = prepair_ngrams(review)

    # coder = Word2Vec(len(review),2, 2)
    # X = torch.tensor([review])
    # out = coder()

# review = build_vocab(data.review.values)
# print(len(review))

# vocab = set(vocab)


# review = map(lambda x: ngrams(x), review)
# for r in review:
#     print(r)
#     exit()
