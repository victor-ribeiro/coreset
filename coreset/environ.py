import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from functools import partial
from configparser import ConfigParser

from coreset.utils import random_sampler
from coreset.lazzy_greed import lazy_greed

load_dotenv()

DATA_HOME = os.environ.get("DATA_HOME")
DATA_HOME = Path(DATA_HOME)
EXPERIMENTS_HOME = os.environ.get("EXPERIMENTS_HOME")
EXPERIMENTS_HOME = Path(EXPERIMENTS_HOME)
RESULTS_HOME = os.environ.get("RESULTS_HOME")
RESULTS_HOME = Path(RESULTS_HOME)

SIZES = [0.01, 0.02, 0.05, 0.10, 0.15, 25]


def load_config():
    curdir = Path(sys.argv[0])
    name = curdir.name
    cfg_file = Path(curdir, "config.ini")
    parser = ConfigParser()
    with open(cfg_file, mode="r") as cfg:
        parser.read_file(cfg)
    print(curdir)
    xp_cfg, ds_cfg = parser["experiment.config"], parser["dataset.info"]
    out_pth = Path(RESULTS_HOME, name)
    outfile = Path(out_pth, xp_cfg["output"])
    if not out_pth.exists():
        out_pth.mkdir(parents=True)

    data_file = Path(DATA_HOME, name, ds_cfg["file_name"])
    columns = (
        ds_cfg["columns"]
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace('"', "")
        .split(",")
    )

    tgt_name = ds_cfg["target"]

    return (outfile, data_file, columns, tgt_name)
