from config.constants import *
import pandas as pd
from src.data.utils import write_file, FooParams
from itertools import islice
import argparse


def make_2class_subdataset(tsv, c1, c2, num_tiles, set):
    input = pd.read_table(tsv)
    res_c1 = []
    res_c2 = []
    iterator = (
        reversed(list(input.itertuples()))
        if set == "training_set"
        else input.itertuples()
    )
    for tile in iterator:
        if tile.tile_label == c1 and len(res_c1) < num_tiles:
            res_c1.append([tile.wsi_name, tile.tile_name, tile.tile_label])
        if tile.tile_label == c2 and len(res_c2) < num_tiles:
            res_c2.append([tile.wsi_name, tile.tile_name, tile.tile_label])
    return res_c1 + res_c2


def make_2class_subdatasets(micron, c1, c2, num_tiles, train_balanced, filename):
    for set in SETS:
        tsv = (
            TILES_TSV / f"{micron}" / f"{filename}_{set}.tsv"
            if filename
            else TILES_TSV / f"{micron}" / f"{set}.tsv"
        )

        rows = make_2class_subdataset(
            tsv=tsv,
            c1=c1,
            c2=c2,
            # num_tiles=num_tiles if balanced and set == "training_set" else float("inf"),
            num_tiles=num_tiles
            if train_balanced and set == "training_set"
            else float("inf"),
            set=set,
        )
        print(f"{set} | {len(rows)}")

        out_tsv = (
            TILES_TSV / f"{micron}" / f"{filename}_{c1}-{c2}_{set}.tsv"
            if filename
            else TILES_TSV / f"{micron}" / f"{c1}-{c2}_{set}.tsv"
        )
        print(f"input file: {tsv}")
        print(f"output file: {out_tsv}")
        write_file(rows, out_tsv, ["wsi_name", "tile_name", "tile_label"])


tiles_dict = {"400": 400, "800": 325, "1200": 140}


def get_subdataset_args(micron, c1, c2, filename, balanced):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--micron", nargs="?", default=micron, type=str)
    parser.add_argument("-c1", "--class_1", nargs="?", default=c1, type=str)
    parser.add_argument("-c2", "--class_2", nargs="?", default=c2, type=str)
    parser.add_argument("-fn", "--filename", nargs="?", default=filename, type=str)
    parser.add_argument("-b", "--train_balanced", nargs="?", default=balanced, type=int)
    return parser.parse_args()


def build_subdataset(
    micron, c1, c2, num_tiles=float("inf"), filename="", train_balanced=0
):
    print("BUILD SUBDATASET I-IV")
    params = get_subdataset_args(micron, c1, c2, filename, train_balanced)
    make_2class_subdatasets(
        params.micron,
        params.class_1,
        params.class_2,
        num_tiles,
        filename=params.filename,
        train_balanced=params.train_balanced,
    )


if __name__ == "__main__":
    build_subdataset("400", "I", "IV", filename="", train_balanced=0, num_tiles=700)
