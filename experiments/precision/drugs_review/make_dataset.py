import pickle
import numpy as np
import pandas as pd


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from pathlib import Path


nltk.download("stopwords")
nltk.download("wordnet")


if __name__ == "__main__":

    data = pd.read_csv(
        "/Users/victor/Documents/projects/coreset/data/drugs_review/drugsComTrain_raw.tsv",
        sep="\t",
        index_col=0,
    )
    data.dropna(axis="index", inplace=True)
    data["rating"] -= 1
    tgt = data.pop("rating").values.astype(int)
    data = data.review.values

    stop_words = stopwords.words("english")
    ds = map(lambda x: x.lower(), data)
    ds = map(word_tokenize, ds)
    ds = map(
        lambda tkns: filter(
            lambda x: not x in stop_words and x.isalnum() and not x.isdigit(), tkns
        ),
        ds,
    )
    ds = map(lambda x: map(WordNetLemmatizer().lemmatize, x), ds)
    ds = map(lambda x: " ".join(x), ds)
    ds = {"features": list(ds), "target": tgt.tolist()}

    print(ds)

    outfile = "/Users/victor/Documents/projects/coreset/data/drugs_review/transformed_drugs_review.pickle"

    with open(outfile, "wb") as file:
        pickle.dump(ds, file, protocol=pickle.HIGHEST_PROTOCOL)
