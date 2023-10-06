import shutil
from config.constants import *
import pandas as pd
import re
from os import listdir
from os.path import isfile
import os
from src.data.utils import delete_folder_if_exist


def count_tiles_slides(micron, filename):
    for set in SETS:
        print(set)
        dataset = (
            TILES_TSV / f"{micron}/{filename}_{set}.tsv"
            if filename
            else TILES_TSV / f"{micron}/{set}.tsv"
        )
        dataset = pd.read_table(dataset)
        stage_1 = len(dataset[dataset["tile_label"] == "I"]["wsi_name"].unique())
        stage_4 = len(dataset[dataset["tile_label"] == "IV"]["wsi_name"].unique())
        print(f"stage 1: {stage_1}\nstage4: {stage_4}")


def wsi_subdataset_labels():
    original = WSI_TSV / "wsi-cordero.tsv"
    original = pd.read_table(original)
    for set in SETS:
        dataset = WSI_TSV / "training_set.tsv"
        dataset = pd.read_table(dataset)
        for s in dataset.itertuples():
            if (
                s.Stage
                != original[original["filename"] == s.filename]["Stage"].values[0]
            ):
                print(f"LABEL ERROR FOR SLIDE {s.filename}")
                exit()


def tiles_labels_test(micron):
    for set in SETS:
        print(f"{set}\n")
        original_wsi = WSI_TSV / f"{set}.tsv"
        original_wsi = pd.read_table(original_wsi)
        mytiles = TILES_TSV / f"{micron}" / f"{set}.tsv"
        mytiles = pd.read_table(mytiles)
        for wsi in original_wsi.itertuples():
            tiles = mytiles[mytiles["wsi_name"] == wsi.filename]
            if len(tiles) == 0:
                print(f"ERROR WSI {wsi.filename} DOES NOT EXIST IN TILES TSV")
                exit()

            tilelabel = tiles["tile_label"].unique()
            print(f"wsi label: {wsi.Stage}")
            print(f"tile label: {tilelabel}")

            if len(tilelabel) != 1 or tilelabel[0] != wsi.Stage:
                print(f"LABEL ERROR ON SLIDE {wsi.filename}")


def filename_test():
    # tsv = WSI_TSV / "labeled.tsv"
    tsv = WSI_TSV / "wsi-cordero.tsv"
    # tsv = WSI_TSV / "no_label.tsv"
    tsv = pd.read_table(tsv)
    tsv = list(tsv["filename"])

    data = WSI_DATA
    onlyfiles = [
        f.replace(".mrxs", "") for f in listdir(data) if isfile(os.path.join(data, f))
    ]
    print(f"tsv = {len(tsv)}")
    print(f"files = {len(onlyfiles)}")
    tf = set(tsv).difference(set(onlyfiles))
    print(f"tsv - files = {tf}")


def get_wsi_names(file):
    df = pd.read_table(file)
    res = []
    wsi = True if "filename" in df else False
    for row in df.itertuples():
        txt = row.filename if wsi else row.tile_name
        x = re.search(r"\d+-\d+_HE", txt)
        res.append(x.group())
    return set(res)


def extraction_test(micron, subset):
    print("WSI NOT EXTRACTED")
    for s in SETS:
        wsi = WSI_TSV / f"sub_{s}.tsv" if subset else WSI_TSV / f"{s}.tsv"
        tiles = (
            TILES_TSV / f"{micron}/sub_{s}.tsv"
            if subset
            else TILES_TSV / f"{micron}/{s}.tsv"
        )
        tiles_names = get_wsi_names(tiles)
        print(tiles_names)
        wsi_names = get_wsi_names(wsi)
        res = list(set(wsi_names) - set(tiles_names))
        print(f"{s} {len(res)} of {len(wsi_names)}")
        print(res)


def tiles_skipped_test():
    wsi = "16-1465_HE.mrxs"
    cp = 1

    # copy tiles
    if cp == 1:
        table = pd.read_table(TILES_TSV / "400/training_set.tsv")
        table = table[table["wsi_name"] == wsi]
        for tile in table.itertuples():
            shutil.copyfile(
                TILES_DATA / f"400/{tile.tile_name}",
                TILES_DATA / f"16-1465/{tile.tile_name}",
            )

    # make skipped dir
    table = TILES_TSV / ""


def copy_skipped_tiles():
    table = pd.read_table(ROOT_DIR / "data/tsv/tiles/400/skipped.tsv")
    delete_folder_if_exist(ROOT_DIR / f"data/skipped_400/")
    for tile in table.itertuples():
        shutil.copyfile(
            ROOT_DIR / f"data/tiles/400/{tile.tile_name}",
            ROOT_DIR / f"data/skipped_400/{tile.tile_name}",
        )


def tiles_skipped_test():
    wsi = "16-1465_HE"
    cp = 0

    # copy tiles
    if cp == 1:
        table = pd.read_table(TILES_TSV / "400/training_set.tsv")
        delete_folder_if_exist(TILES_DATA / "16-1465")
        for tile in table.itertuples():
            if wsi in tile.tile_name:
                print(f"copy {tile.tile_name}")
                shutil.copyfile(
                    TILES_DATA / f"400/{tile.tile_name}",
                    TILES_DATA / f"16-1465/{tile.tile_name}",
                )

    # make skipped dir
    table = pd.read_table(ROOT_DIR / "data/tsv/tiles/400/training_set_skipped.tsv")
    delete_folder_if_exist(TILES_DATA / "skipped_16-1465")
    for tile in table.itertuples():
        if wsi in tile.tile_name:
            print(f"copy skipped tile {tile.tile_name}")
            shutil.copyfile(
                ROOT_DIR / f"data/tiles/400/{tile.tile_name}",
                TILES_DATA / f"skipped_16-1465/{tile.tile_name}",
            )


if __name__ == "__main__":
    count_tiles_slides("400", "I-IV")
