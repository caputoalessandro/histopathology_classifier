import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from config.constants import *
import params as pp
import argparse
import re
from src.data.utils import split_test_train


class FooParams(pp.Params):
    micron = None
    model = None
    function = None
    metric = None
    sets = None
    title = None
    tiles_type = None
    binary = None
    subset = None
    preprocessing = None
    score = None
    num_classes = None
    lh = None
    tsv_filename = None
    I_IV = None
    mode = None
    color = None
    plot_title = None
    max = None
    metrics = None


def get_visualize_args(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--micron", nargs="?", default=params.micron, type=str)
    parser.add_argument("-t", "--title", nargs="?", default=params.title, type=str)
    parser.add_argument(
        "-f", "--function", nargs="?", default=params.function, type=str
    )
    parser.add_argument(
        "-tt", "--tiles_type", nargs="?", default=params.tiles_type, type=str
    )
    params.update(vars(parser.parse_args()))


def count_stages_istances(df_name, df, params):
    d = {}
    for s in ["I", "II", "III", "IV"]:
        count = len(df[df["Stage"] == s])
        d.setdefault(s, count)

    if params.mode == "lh":
        res = {"I - II": d["I"] + d["II"], "III - IV": d["III"] + d["IV"]}
    elif params.mode == "I_IV":
        res = {"I": d["I"], "IV": d["IV"]}
    elif params.mode == "all":
        res = d

    return {df_name: list(res.values())}


def get_split_stats(file, params):
    count_dict = {}
    training_set, test_set = split_test_train(file)
    dfs = {"training_set": training_set, "test_set": test_set}

    whole_ds = pd.read_table(WSI_TSV / "labeled.tsv")
    res = count_stages_istances("whole_dataset", whole_ds, params)
    count_dict.update(res)

    for set in SETS:
        _, df = dfs[set]
        res = count_stages_istances(set, df, params)
        count_dict.update(res)

    return pd.DataFrame.from_dict(count_dict)


def print_wsi(img):
    figure(figsize=(5, 6), dpi=400)
    plt.imshow(img)
    plt.show(ROOT_DIR / "assets")
