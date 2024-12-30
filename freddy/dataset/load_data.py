import re
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, minmax_scale, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from freddy.dataset.utils import hash_encoding, split_dataset

datasets = {}


def _register(name):
    def deco(f_):
        datasets[name] = f_
        return f_

    return deco


@_register("adult")
def _load_adult(fp):
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    tgt_name = "label"
    data = pd.read_csv(fp, engine="pyarrow", names=names)
    data.replace(" ?", np.nan, inplace=True)
    data[tgt_name] = data[tgt_name].map({" >50K": 1, " <=50K": 0})
    data = pd.get_dummies(
        data,
        columns=["sex", "education", "race", "relationship", "workclass"],
        drop_first=False,
    )
    data = hash_encoding(
        data, "native-country", "occupation", "marital-status", n_features=5
    )
    return data


@_register("bike_share")
def _load_bike_share(fp):
    names = [
        "dteday",
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt",
    ]

    # data = pd.read_csv(fp, engine="pyarrow", names=names)
    data = pd.read_csv(fp, engine="pyarrow")
    data["dteday"] = OrdinalEncoder().fit_transform(
        data["dteday"].values.reshape(-1, 1)
    )
    data = pd.get_dummies(
        data,
        columns=["workingday", "holiday", "weathersit"],
        drop_first=False,
    ).astype(float)
    data.loc[:, ["casual", "registered"]] = minmax_scale(
        data.loc[:, ["casual", "registered"]]
    )
    return data


@_register("covtype")
def _load_covtype(fp):
    names = [
        "elevation",
        "aspect",
        "slope",
        "horizontal_distance_to_hydrology",
        "vertical_distance_to_hydrology",
        "horizontal_distance_to_roadways",
        "hillshade_9am",
        "hillshade_noon",
        "hillshade_3pm",
        "horizontal_distance_to_fire_points",
        "wilderness_area_0",
        "wilderness_area_1",
        "wilderness_area_2",
        "wilderness_area_3",
        "soil_type_0",
        "soil_type_1",
        "soil_type_2",
        "soil_type_3",
        "soil_type_4",
        "soil_type_5",
        "soil_type_6",
        "soil_type_7",
        "soil_type_8",
        "soil_type_9",
        "soil_type_10",
        "soil_type_11",
        "soil_type_12",
        "soil_type_13",
        "soil_type_14",
        "soil_type_15",
        "soil_type_16",
        "soil_type_17",
        "soil_type_18",
        "soil_type_19",
        "soil_type_20",
        "soil_type_21",
        "soil_type_22",
        "soil_type_23",
        "soil_type_24",
        "soil_type_25",
        "soil_type_26",
        "soil_type_27",
        "soil_type_28",
        "soil_type_29",
        "soil_type_30",
        "soil_type_31",
        "soil_type_32",
        "soil_type_33",
        "soil_type_34",
        "soil_type_35",
        "soil_type_36",
        "soil_type_37",
        "soil_type_38",
        "soil_type_39",
        "cover_type",
    ]
    tgt_name = "cover_type"
    data = pd.read_csv(fp, engine="pyarrow", names=names)
    data[tgt_name] = data[tgt_name] - 1
    return data


@_register("drugs_review")
def _load_drugs_review(fp):

    def clean_sent(sent, sub_pattern=r"[\W\s]+"):
        sent = sent.lower()
        sent = re.sub(sub_pattern, " ", sent)
        sent = re.split(r"\W", sent)
        sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
        return sent

    tgt_name = "rating"
    with open(fp, "rb") as file:
        data = pickle.load(file)
    data, tgt = data["features"], data["target"]
    data = map(clean_sent, data)

    # data = CountVectorizer(max_features=1500).fit_transform(data).toarray()
    data = (
        TfidfVectorizer(max_features=1500, min_df=0.05, max_df=0.98)
        .fit_transform(data)
        .toarray()
    )
    data = PCA(n_components=100).fit_transform(data)
    data = pd.DataFrame(data=data)
    data[tgt_name] = [*map(int, tgt)]
    data[tgt_name] = data[tgt_name].map(lambda x: 1 if x > 5 else 0)
    data.columns = data.columns.astype(str)
    return data


@_register("nursery")
def _load_nursery(fp):
    names = [
        "parents",
        "has_nurs",
        "form",
        "children",
        "housing",
        "finance",
        "social",
        "health",
        "label",
    ]
    tgt_name = "label"
    data = pd.read_csv(fp, engine="pyarrow", names=names)
    data["children"] = data["children"].map({"1": 1, "2": 2, "3": 3, "more": 4})
    data[tgt_name] = data[tgt_name].map(
        lambda x: "recommend" if x == "very_recom" or x == "priority" else x
    )
    data[tgt_name] = LabelEncoder().fit_transform(data[tgt_name]).astype(int)

    for col in names[4:]:
        tmp = data[col].values.reshape(-1, 1)
        data[col] = OrdinalEncoder().fit_transform(tmp)
    data = pd.get_dummies(
        data,
        columns=["parents", "has_nurs", "form"],
        drop_first=False,
    )
    return data


@_register("sgemm")
def _load_sgemm(fp):
    avg_names = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]
    tgt_name = "run_avg"
    data = pd.read_csv(fp, engine="pyarrow", index_col=0, skiprows=1)

    data[tgt_name] = data[avg_names].mean(axis=1)
    data = data.drop(columns=avg_names)
    return data


def load(name, fp, label, test_size=None):
    _load_fn = datasets[name]
    print(f"[READING] {name} dataset: {fp}")
    data = _load_fn(fp)
    n, c = data.shape
    print(f"register:\t{n}\ncols:\t{c}")
    if not test_size:
        return data
    print(f"[READING] {name} dataset - ok")
    return split_dataset(data, label, test_size)
