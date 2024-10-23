import pickle
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.under_sampling import RandomUnderSampler


nltk.download("stopwords")
nltk.download("wordnet")


if __name__ == "__main__":

    data = pd.read_csv(
        # "/Users/victor/Documents/projects/coreset/data/drugs_review/drugsComTest_raw.tsv",
        "/Users/victor/Documents/projects/coreset/data/drugs_review/drugsComTrain_raw.tsv",
        sep="\t",
        index_col=0,
    )
    data.dropna(axis="index", inplace=True)
    data["rating"] -= 1
    tgt = data.pop("rating").values.astype(int)
    dnames = data["drugName"].unique().tolist()
    condition = data["condition"].unique().tolist()
    data = data.review.values

    stop_words = stopwords.words("english")

    stop_words += dnames
    stop_words += condition
    ds = map(lambda x: x.lower(), data)
    ds = map(word_tokenize, ds)
    ds = map(
        lambda tkns: filter(
            lambda x: not x in stop_words and x.isalnum() and not x.isdigit(), tkns
        ),
        ds,
    )

    # ds = map(lambda x: map(PorterStemmer().stem, x), ds)
    ds = map(lambda x: map(WordNetLemmatizer().lemmatize, x), ds)
    ds = map(lambda x: " ".join(x), ds)

    ds = map(lambda x: re.sub(r"[0-9]+[a-z]{1,2}", "", x), ds)
    ds = map(lambda x: re.sub(r"[\s]+", " ", x), ds)
    # ds = {"features": list(ds), "target": tgt.tolist()}

    features, target = RandomUnderSampler().fit_resample(list(ds), tgt.tolist())
    ds = {"features": features, "target": target}

    print(ds["features"])

    outfile = "/Users/victor/Documents/projects/coreset/data/drugs_review/transformed_drugs_review.pickle"

    with open(outfile, "wb") as file:
        pickle.dump(ds, file, protocol=pickle.HIGHEST_PROTOCOL)
