from openslide import open_slide
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from histolab.slide import Slide
from histolab.tiler import GridTiler
from config.constants import *
import argparse
import csv
import params as pp
from PIL import Image
import shutil
import pprint


class FooParams(pp.Params):
    toy_flag = None
    micron = None
    memory_factor = None
    class_1 = (None,)
    class_2 = None
    num_tiles = None
    score = None
    cpu = None


def write_wsi_names():
    for s, wsi_names in split_test_train(WSI_TSV / "labeled.tsv"):
        rows = wsi_names[["filename", "Stage"]].values.tolist()
        rows = [wsi for wsi in rows]
        write_file(
            rows,
            WSI_TSV / f"{s}.tsv",
            ["filename", "Stage"],
        )


def write_file(rows, path, first_row, delim="\t"):
    with open(path, "w") as f:
        tsv_writer = csv.writer(f, delimiter=delim)
        tsv_writer.writerow(first_row)
        tsv_writer.writerows(rows)
    # print(f"file writed: {path}")


def write_row(row, path, mode):
    with open(path, mode) as f:
        tsv_writer = csv.writer(f, delimiter="\t")
        tsv_writer.writerow(row)
        f.close()


def get_mpp():
    slide = open_slide(WSI_DATA / "16-1851_HE.mrxs")
    return float(slide.properties.get("openslide.mpp-x"))


# def split_test_train(file):
#     tsv = WSI_TSV / file
#     data = pd.read_table(tsv)

#     training_set, test_val_set = train_test_split(
#         data, train_size=0.6, test_size=0.4, random_state=42, stratify=data[["Stage"]]
#     )
#     # test_set, validation_set = train_test_split(
#     #     test_val_set,
#     #     train_size=0.5,
#     #     test_size=0.5,
#     #     random_state=42,
#     #     stratify=test_val_set[["Stage"]],
#     # )

#     res = [
#         ("training_set", training_set),
#         ("test_set", test_set),
#         #("validation_set", validation_set),
#     ]

#     for s, wsi_names in res:
#         rows = wsi_names[["filename", "Stage"]].values.tolist()
#         rows = [wsi for wsi in rows]
#         write_file(
#             rows,
#             WSI_TSV / f"{s}.tsv",
#             ["filename", "Stage"],
#         )

#     return res


def split_test_train(file):
    tsv = WSI_TSV / file
    data = pd.read_table(tsv)

    training_set, test_set = train_test_split(
        data, train_size=0.6, test_size=0.4, random_state=43, stratify=data[["Stage"]]
    )
    
    res = [
        ("training_set", training_set),
        ("test_set", test_set),
    ]

    for s, wsi_names in res:
        rows = wsi_names[["filename", "Stage"]].values.tolist()
        rows = [wsi for wsi in rows]
        write_file(
            rows,
            WSI_TSV / f"{s}.tsv",
            ["filename", "Stage"],
        )

    return res


def delete_folder_if_exist(folder):
    if folder.is_dir():
        shutil.rmtree(folder)
        print(f"{folder} removed")
    folder.mkdir(parents=True, exist_ok=True)


def delete_file_if_exist(file):
    if file.is_file():
        file.unlink()
        print(f"{file} removed")
    file.parent.mkdir(parents=True, exist_ok=True)


def read_slide():
    slide_path = ROOT_DIR / "data/wsi/16-1851_HE"
    slide = Slide(slide_path, ROOT_DIR / "data", use_largeimage=True)

    random_tiles_extractor = GridTiler(
        tile_size=(3000, 3000),
        level=2,
        check_tissue=True,  # default
        tissue_percent=80.0,  # default
        prefix="prova/we",
        # save tiles in the "grid" subdirectory of slide's processed_path
        suffix=".png",  # default
    )
    random_tiles_extractor.extract(slide)


def save_wsi_thubnails():
    for wsi in WSI_DATA.iterdir():
        out = ROOT_DIR / "data/thumbnails"
        out.mkdir(exist_ok=True)
        print(wsi.stem)
        if wsi.is_file():
            slide = Slide(wsi, ROOT_DIR / "data/thumbnails")
            # slide.scaled_image(scale_factor=32).save(out / f"{wsi.stem}.png", "PNG")
            pprint.pprint(slide.properties)
            break


if __name__ == "__main__":
    for k in ["tr", "te"]:
        first = pd.read_csv(ROOT_DIR / f"400_{k}_23.csv")
        second = pd.read_csv(ROOT_DIR / f"400_{k}_other.csv")
        res = pd.merge(first, second)
        res.to_csv(ROOT_DIR / f"tsv/wandb/400_{k}_other.csv", index=False)