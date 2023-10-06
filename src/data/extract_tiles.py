import pandas as pd
from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler
from multiprocessing import Pool, cpu_count
import random
import csv
import math
from config.constants import *
from src.data.utils import get_mpp, delete_file_if_exist
import argparse


def extract_grid(slide, micron, wsi_name, tile_size, toy):
    print(f"{wsi_name} start")

    if toy:
        print("TOY FLAG ON")
        random_tiles_extractor = RandomTiler(
            tile_size=tile_size,
            n_tiles=2,
            level=2,
            check_tissue=True,  # default
            tissue_percent=80.0,  # default
            prefix=wsi_name
            + str(
                random.randint(0, 900)
            ),  # save tiles in the "grid" subdirectory of slide's processed_path
            suffix=".png",  # default
        )
        r = random_tiles_extractor.locate_tiles(
            slide=slide,
            scale_factor=32,
            alpha=32,
            outline="#046C4C",
        )
        Path(f"data/tiles/{micron}/exgr").mkdir(parents=True, exist_ok=True)
        r.save(ROOT_DIR / f"data/tiles/{micron}/exgr/{str(random.randint(0, 900))}.png")
        random_tiles_extractor.extract(slide)

    else:
        grid_tiles_extractor = GridTiler(
            tile_size=tile_size,
            level=0,
            check_tissue=True,  # default
            pixel_overlap=0,  # default
            prefix=wsi_name,  # save tiles in the "grid" subdirectory of slide's processed_path
            tissue_percent=80.0,
            suffix=".png",  # default
        )
        grid_tiles_extractor.extract(slide)
    print(f"{wsi_name} finish")


def extract_from_slide(args):
    wsi_name, micron, tile_size, out_folder, skipped_slides, toy = args
    wsi_path = WSI_DATA / (wsi_name + ".mrxs")
    slide = Slide(wsi_path, out_folder)
    try:
        extract_grid(slide, micron, wsi_path.stem + "_", tile_size, toy)
    except Exception as e:
        print(f"EXTRACTION ERROR ON SLIDE {wsi_name}")
        print(e)
        with open(skipped_slides, "a") as f:
            writer = csv.writer(f)
            writer.writerow([wsi_name])


def extract_from_datasets(tile_size, skipped_slides, file, params):
    tiles_folder = TILES_DATA / f"{params.micron}_new"
    wsi_dataframe = pd.read_table(file)
    
    to_map = [
        (
            wsi_name,
            params.micron,
            tile_size,
            tiles_folder,
            skipped_slides,
            params.toy_flag,
        )
        for wsi_name in wsi_dataframe["filename"]
    ]

    chunk = math.ceil(len(to_map) / params.cpu)
    print(f"chunk_size = {chunk}")

    pool = Pool(params.cpu)
    pool.map(extract_from_slide, to_map)


def get_extraxt_tiles_args(micron, toy_flag, cpu):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--micron", nargs="?", default=str(micron), type=str)
    parser.add_argument("-tf", "--toy_flag", nargs="?", default=toy_flag, type=int)
    parser.add_argument("-c", "--cpu", nargs="?", default=cpu, type=int)
    return parser.parse_args()


def main():
    params = get_extraxt_tiles_args(micron="1200", toy_flag=0, cpu=cpu_count())
    skipped_slides = TILES_TSV / f"{params.micron}" / "skipped_slides.tsv"
    delete_file_if_exist(skipped_slides)
    wsi_file = WSI_TSV / ("labeled_toy.tsv" if params.toy_flag else "labeled.tsv")
    mpp = get_mpp()
    tile_size = int(int(params.micron) / mpp)
    extract_from_datasets((tile_size, tile_size), skipped_slides, wsi_file, params)


if __name__ == "__main__":
    main()
