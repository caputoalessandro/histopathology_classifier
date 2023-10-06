from histolab.tile import Tile
from histolab.scorer import CellularityScorer, NucleiScorer
from config.constants import *
from PIL import Image, ImageFile
import pandas as pd
from src.data.utils import write_row, delete_file_if_exist
import params as pp
from wsi_tile_cleanup import filters, utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FooParams(pp.Params):
    micron = None
    type = None
    thr = None
    score_calc = None
    in_tsv = None


def thr_check(score, thr, type):
    if type == "black":
        return score < thr
    else:
        return score > thr


def get_black_perc(tile_path):
    vi_tile = utils.read_image(tile_path)
    bands = utils.split_rgb(vi_tile)
    try:
        black_perc = filters.blackish_percent(bands)
    except Exception as e:
        print(f"error on tile {tile_path}")
        black_perc = None
    return black_perc


def make_thresholded_tsv(
    micron, stype, thr, on_test_set, filename, sfilename, sets=SETS
):
    print(f"BUILD {stype.upper()} SCORED TSV | THR={thr}")
    for set in sets:
        in_tsv = (
            TILES_TSV / f"{micron}/{filename}_{set}.tsv"
            if filename != ""
            else TILES_TSV / f"{micron}/{set}.tsv"
        )
        out_tsv = (
            TILES_TSV / micron / f"scored_{stype}_{thr}_{filename}_{set}.tsv"
            if filename != ""
            else TILES_TSV / micron / f"scored_{stype}_{thr}_{set}.tsv"
        )
        delete_file_if_exist(out_tsv)
        print(f"input: {in_tsv}")
        print(f"output: {out_tsv}")
        in_tsv = pd.read_table(in_tsv)

        score_tsv = (
            TILES_TSV / micron / f"scored_{stype}_{sfilename}_{set}.tsv"
            if sfilename != ""
            else TILES_TSV / micron / f"scored_{stype}.tsv"
        )
        score_tsv = pd.read_table(score_tsv)
        in_tsv = pd.merge(
            in_tsv, score_tsv, how="inner", on=["wsi_name", "tile_name", "tile_label"]
        )

        write_row(["wsi_name", "tile_name", "tile_label"], out_tsv, "a")
        counter = 0

        for i, tile in enumerate(in_tsv.itertuples()):
            if set == "test_set" and not on_test_set:
                row = [tile.wsi_name, tile.tile_name, tile.tile_label]
                write_row(row, out_tsv, "a")
                counter += 1

            else:
                if thr_check(tile.score, thr, stype):
                    row = [tile.wsi_name, tile.tile_name, tile.tile_label]
                    write_row(row, out_tsv, "a")
                    counter += 1

        print(f"{set} size | {counter} of {len(in_tsv)}")


def get_score(tile_path, type):
    try:
        img = Image.open(tile_path).convert("RGB")
    except Exception as e:
        print(f"READ ERROR ON TILE {tile_path}")
        print(e)
        return None

    tile = Tile(img, (0, 0, 0, 0))
    s = None
    if type == "cell":
        scorer = CellularityScorer()
        s = scorer(tile)
    if type == "nuclei":
        scorer = NucleiScorer()
        s = scorer(tile)
    if type == "black":
        s = get_black_perc(tile_path)
    return s


def build_scored_tsv(micron, stype, filename, data_path, sets=SETS):
    for set in sets:
        tsv = TILES_TSV / f"{micron}/{filename}_{set}.tsv"
        out_tsv = TILES_TSV / f"{micron}/scored_{stype}_{filename}_{set}.tsv"

        table = pd.read_table(tsv)
        tiles = list(enumerate(table.itertuples()))
        skip_count = 0

        if out_tsv.exists():
            out_len = len(pd.read_table(out_tsv))
            if out_len > 0:
                print(
                    f"output file is not empty, process starting from tile: {out_len}"
                )
                tiles = tiles[out_len:]
        else:
            print(f"{out_tsv} not exist, starting from new file")
            write_row(
                ["wsi_name", "tile_name", "tile_label", "score"], out_tsv, mode="a"
            )

        for i, tile in tiles:
            score = get_score(data_path / tile.tile_name, stype)
            if score:
                print(f"{set}: {i} of {len(table)} calculated | {score}", end="\r")
                write_row(
                    [tile.wsi_name, tile.tile_name, tile.tile_label, score],
                    out_tsv,
                    "a",
                )
            else:
                skip_count += 1

        print(f"procedd finished, skipped tiles: {skip_count}")


def concat_scored_files(micron, score):
    train = pd.read_table(TILES_TSV / f"{micron}/scored_{score}_training_set.tsv")
    test = pd.read_table(TILES_TSV / f"{micron}/scored_{score}_test_set.tsv")
    res = pd.concat([train, test])
    res.to_csv(TILES_TSV / f"{micron}/scored_{score}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    make_thresholded_tsv(
        micron="400",
        stype="cell",
        thr=0.1,
        on_test_set=1,
        filename="scored_nuclei_0.1_scored_black_0.2",
        sfilename="",
    )
    # concat_scored_files(400, "cell")
