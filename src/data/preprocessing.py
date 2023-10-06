import pandas as pd
from src.data.stain_normalization import stain_normalization
import numpy as np
import cv2
from matplotlib import pyplot as plt
from config.constants import TILES_DATA, ROOT_DIR, TILES_TSV, SETS
import torchstain
from torchvision import transforms
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
from src.data.utils import delete_folder_if_exist

"""
1. denoising
2. color normalization
3. contrast equalization
4. stain normalization
5.
"""
resize_table = {"50": 100, "100": 365, "400": 1460, "800": 1800, "1200": 2191}

def get_transform(size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(size, size), antialias=True),
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


def print_histogram(img, filename):
    # histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color="b")
    plt.hist(img.flatten(), 256, [0, 256], color="r")
    plt.xlim([0, 256])
    plt.legend(("cdf", "histogram"), loc="upper left")
    plt.savefig(TILES_DATA / f"100/output/h/{filename}_histogram.png")
    cv2.imwrite(str(TILES_DATA / f"100/output/{filename}.png"), img)


def CLAHE_colored(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r = clahe.apply(img[:, :, 0])
    g = clahe.apply(img[:, :, 1])
    b = clahe.apply(img[:, :, 2])
    colorimage_clahe = np.stack((b, g, r), axis=2)
    return colorimage_clahe


def CLAHE_gray(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return clahe.apply(img)


def over_mean_cut(img, t):
    mean = np.mean(img, axis=tuple(range(img.ndim - 1)))
    img[np.all(img > mean, axis=-1)] = mean
    return img


def preprocessing(args):
    image_path, micron = args
    img = cv2.imread(str(image_path), cv2.COLOR_BGR2RGB)
    # print_histogram(img, "input")

    # img = over_mean_cut(img, 135)
    # # print_histogram(img, "changed")

    img = CLAHE_colored(img)
    print_histogram(img, "heq")
    print(f"{image_path} preprocessed")

    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # print_histogram(img, "denoised")

    # # STAIN NORMALIZATION
    # T = get_transform(resize_table[micron])
    # normalizer = get_normalizer(T)
    # img = stain_normalization(str(image_path), T, normalizer)
    # # print_histogram(img, "stain")

    cv2.imwrite(str(TILES_DATA / f"{micron}_preprocessed/{image_path.stem}.png"), img)


def main(micron, filename, data_folder, sets=SETS):
    delete_folder_if_exist(TILES_DATA / f"{micron}_preprocessed")
    for set in sets:
        print(set.upper())
        tiles = pd.read_table(TILES_TSV / f"{micron}/{filename}_{set}.tsv")
        to_map = [
            (TILES_DATA / f"{data_folder}/{tile.tile_name}", micron)
            for tile in tiles.itertuples()
        ]
        pool = Pool(cpu_count())
        for _ in tqdm(pool.imap(preprocessing, to_map), total=len(to_map)):
            pass


if __name__ == "__main__":
    # img_path = TILES_DATA / "100/16-1604_HE_tile_10003_level0_57305-39420-57670-39785.png"
    img_path = ROOT_DIR / "data/normalization/400/15-665_HE_tile_91_level0_5840-45260-7300-46720.png"
    preprocessing([img_path, "400"])
    #main("400", "scored_cell_0.1_scored_black_0.2_I-IV", data_folder="400")

