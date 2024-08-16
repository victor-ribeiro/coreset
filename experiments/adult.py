import os

print(os.environ.get("DATA_HOME"))
# import pandas as pd
# import numpy as np

# from pathlib import Path

# from sklearn.metrics import precision_score, balanced_accuracy_score, accuracy_score


# def map_vals(mapping, column):
#     def _inner(data):
#         dataset = data.copy()
#         dataset[column] = dataset[column].map(mapping).astype(int)
#         return dataset

#     return _inner


# def transform_wrap(func, tgt_name):
#     def inner(data):
#         dataset = data.copy()
#         tgt = dataset[tgt_name].astype(int)
#         names = dataset.columns.astype(str)
#         dataset = pd.DataFrame(func(dataset), columns=names)
#         dataset[tgt_name] = tgt.values
#         return dataset

#     return inner


# def one_h_coding(*names):
#     def _inner(data):
#         return pd.get_dummies(data, columns=list(names), drop_first=False)

#     return _inner


# def hash_encoding(hash_names, n_features=15):
#     def _inner(data):
#         dataset = data.copy()
#         corpus = dataset[hash_names].apply(lambda x: "".join(x), axis="columns")

#         encoder = HashingVectorizer(n_features=n_features)
#         encoded = encoder.fit_transform(corpus).toarray()
#         encoded = pd.DataFrame(encoded)
#         dataset.drop(columns=hash_names, inplace=True)
#         dataset = pd.concat([dataset, encoded], axis="columns")
#         dataset.columns = dataset.columns.astype(str)
#         return dataset

#     return _inner


# if __name__ == "__main__":
#     data_path = Path(data_root, "adult", "adult.data")
#     dataset = pd.read_csv(data_path, names=config["COLUMNS"])
#     dataset.drop(columns="education", inplace=True)
#     dataset = dataset.replace(" ?", np.nan)
#     dataset = dataset.dropna(axis=0)
#     base_preprocessing = pipeline(
#         # clean
#         map_vals({" >50K": 1, " <=50K": 0}, config["tgt_name"]),
#         # handle_imb(config["tgt_name"]),
#         input_vals("most_frequent"),
#         # encoding
#         one_h_coding("sex"),
#         hash_encoding(config["HASH_NAMES"], n_features=2),
#         transform_wrap(normalize, config["tgt_name"]),
#     )
