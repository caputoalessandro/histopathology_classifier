from torchvision import transforms
from multiprocessing import cpu_count, Pool
import math
from config.constants import *
import argparse
import torchstain
import cv2
import torch
from src.data.utils import write_file, delete_folder_if_exist
import shutil
from wsi_tile_cleanup import filters, utils

target = {
    "400": ROOT_DIR
    / "data/normalization/400/15-665_HE_tile_91_level0_5840-45260-7300-46720.png",
    "800": ROOT_DIR
    / "data/normalization/800/18-3194_HE_tile_29_level0_17526-61779-20447-64700.png",
}


def get_preprocessing_args(micron):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--micron", nargs="?", default=str(micron), type=str)
    return parser.parse_args()


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(1460, 1460), antialias=True),
            transforms.Normalize(mean=0.0, std=(1 / 255.0, 1 / 255.0, 1 / 255.0)),
        ]
    )


def get_normalizer(transform):
    TARGET_PATH = str(
        ROOT_DIR
        / "data/normalization/400/15-665_HE_tile_91_level0_5840-45260-7300-46720.png"
    )
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    target = cv2.cvtColor(cv2.imread(TARGET_PATH), cv2.COLOR_BGR2RGB)
    normalizer.fit(transform(target))
    return normalizer


def stain_normalization(tile_path, T, normalizer):
    to_transform = cv2.cvtColor(cv2.imread(tile_path), cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)
    prep_tile, H, E = normalizer.normalize(I=t_to_transform, stains=True)
    return prep_tile.numpy()


def cut_black_or_white_tiles(tile_path):
    vi_tile = utils.read_image(tile_path)
    bands = utils.split_rgb(vi_tile)
    black_perc = filters.blackish_percent(bands)
    if black_perc > 0.2:
        print(tile_path)
        print(f"blackish: {black_perc}")
        return True
    else:
        return False


def preprocess_tile(tile_path, T, normalizer, out_dir, stain):
    print(tile_path)
    if not cut_black_or_white_tiles(tile_path):
        p = Path(f"{out_dir}/{Path(tile_path).name}")
        if stain:
            prep_tile = stain_normalization(tile_path, T, normalizer)
            cv2.imwrite(p, prep_tile.numpy())
        else:
            shutil.copyfile(ROOT_DIR / tile_path, p)
    else:
        return [Path(tile_path).name]


def preprocess_tiles(tiles_paths, transform, normalizer, out_dir, stain):
    cpu = cpu_count()
    cpu_to_use = math.ceil(cpu * 0.75)
    chunk: int = math.ceil(len(tiles_paths) / cpu_to_use)

    to_map = [
        (str(tile_path), transform, normalizer, str(out_dir), stain)
        for tile_path in tiles_paths
    ]
    torch.set_num_threads(1)

    with Pool(cpu_to_use) as p:
        skipped = [
            res for res in p.starmap(preprocess_tile, to_map, chunksize=chunk) if res
        ]

    write_file(skipped, TILES_TSV / "400/skipped.tsv", ["tile_name"])


def save_preprocessed_tiles(tiles_dir, out_dir, stain):
    tiles_paths = [tile_path for tile_path in tiles_dir.iterdir()]
    # tiles_paths = [TILES_DATA / f"400/{t.tile_name}" for t in pd.read_table(TILES_TSV / "400/16-1465_HE.tsv").itertuples()]
    T = get_transform()
    normalizer = get_normalizer(T)
    preprocess_tiles(tiles_paths, T, normalizer, out_dir, stain)


def preprocess():
    # args = get_preprocessing_args(100)
    path = TILES_DATA / "400"
    out_path = path.parent / "preprocessed/400"
    delete_folder_if_exist(out_path)
    save_preprocessed_tiles(tiles_dir=path, out_dir=out_path, stain=False)


if __name__ == "__main__":
    preprocess()
