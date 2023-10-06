import re
import pandas as pd
from src.data.utils import write_file, delete_file_if_exist
from config.constants import SETS, WSI_TSV, TILES_TSV
from utils import FooParams, split_test_train
from config.constants import TILES_DATA


def get_tile_label(tile_name, wsi_file):
    df_file = wsi_file.loc[wsi_file["filename"] == tile_name]
    tile_label = df_file["Stage"].iloc[0]
    return tile_label


def make_dataset_dict(datasets):
    d = {}
    for set_name, dataset in datasets:
        for wsi in dataset.itertuples():
            d.update({wsi.filename: set_name})
    return d


def make_tiles_tsv_dict(datasets, tiles_dir):
    res = {"training_set": [],"validation_set": [] ,"test_set": []}
    wsi_dict = make_dataset_dict(datasets)
    wsi_file = pd.read_table(WSI_TSV / "labeled.tsv")
    tiles = list(enumerate(sorted(tiles_dir.iterdir())))
    for n, tile in tiles:
        print(f"{n} of {len(tiles)} executed", end="\r")
        tile_wsi_name = re.match("\d+-\d+_HE", tile.name).group()
        set_dest = wsi_dict[tile_wsi_name]
        label = get_tile_label(tile_wsi_name, wsi_file)
        res[set_dest].append([tile_wsi_name, tile.name, label])
    return res


def make_tiles_tsv_file(micron, dataset, tiles_dir, output_filename):
    to_write = make_tiles_tsv_dict(dataset, tiles_dir)
    for set in SETS:
        tsv_tiles_path = (
            TILES_TSV / micron / f"{output_filename}_{set}.tsv"
            if output_filename != ""
            else TILES_TSV / micron / f"{set}.tsv"
        )
        delete_file_if_exist(tsv_tiles_path)
        rows = to_write[set]
        print(f"{set} size: {len(rows)}")
        write_file(rows, tsv_tiles_path, ["wsi_name", "tile_name", "tile_label"])


def attach_wsi_column(micron, filename):
    for set in SETS:
        to_write = []
        tiles = pd.read_table(TILES_TSV / micron / f"{filename}_{set}.tsv")
        for n, tile in enumerate(tiles.itertuples()):
            print(f"{n} of {len(tiles)} executed", end="\r")
            tile_wsi_name = re.match("\d+-\d+_HE", tile.tile_name).group()
            to_write.append(
                [tile_wsi_name, tile.tile_name, tile.tile_label, tile.score]
            )
        write_file(
            to_write,
            TILES_TSV / micron / f"{filename}_{set}_new.tsv",
            ["wsi_name", "tile_name", "tile_label", "score"],
        )


def main(micron, output_filename):
    wsi_file = WSI_TSV / "labeled.tsv"
    tiles_folder = TILES_DATA / micron
    datasets = split_test_train(wsi_file)
    make_tiles_tsv_file(micron, datasets, tiles_folder, output_filename)
    # attach_wsi_column("100", "scored_nuclei")


if __name__ == "__main__":
    #attach_wsi_column("400","")
    main("800", "")

