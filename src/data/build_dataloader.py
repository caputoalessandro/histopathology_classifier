import pandas as pd
from src.data.build_dataset import EIDOSTilesDataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from config.constants import TILES_DATA, TILES_TSV, SETS, WSI_TSV
from src.data.utils import write_file
import re


def set_dataset_paths(params, sets, filename):
    params.data_path = TILES_DATA / params.micron
    tsv = {set: f"{filename}_{set}.tsv" for set in sets}
    params.tsv.update(tsv)


def get_tile_dataloader(tsv_path, data_path, batch_size, num_classes, transform=None):
    data = EIDOSTilesDataset(tsv_path, data_path, num_classes, transform)
    return DataLoader(
        data,
        num_workers=int(cpu_count() / 2),
        batch_size=batch_size,
        shuffle=True,
    )


def get_tiles_dataloaders(params, transform, sets):
    res = []
    dataloader = None
    for set in sets:
        tsv_name = (
            f"{set}.tsv"
            if params.tsv_filename == ""
            else f"{params.tsv_filename}_{set}.tsv"
        )
        tsv = TILES_TSV / params.micron / tsv_name
        params.tsv.update({set: tsv})
        dataloader = get_tile_dataloader(
            tsv, params.data_path, params.batch_size, params.num_classes, transform
        )
        res.append(dataloader)
    return dataloader if len(sets) == 1 else (r for r in res)


def write_wsi_tiles_file(sets, micron, filename):
    for set in sets:
        tsv = pd.read_table(TILES_TSV / micron / f"{filename}_{set}.tsv")
        tiles_table = pd.read_table(tsv)
        for wsi_name in tiles_table["wsi_name"].unique().tolist():
            to_write = tiles_table[tiles_table["wsi_name"] == wsi_name].tolist()
            write_file(
                to_write,
                TILES_TSV / micron / f"{wsi_name}.tsv",
                ["wsi_name", "tile_name", "tile_label"],
            )
