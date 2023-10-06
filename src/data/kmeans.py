from config.constants import TILES_TSV, TILES_DATA, SETS
import pandas as pd
from sklearn.cluster import KMeans
from src.models.save_model_features import (
    get_tiles_wsi_dataloaders,
    write_tiles_wsi_tsv,
)
import argparse
from torchvision import transforms
import cv2
import numpy as np
from collections import Counter


def get_transfrom(size):
    return transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


resize_table = {"50": 100, "100": 365, "400": 1460, "800": 1800, "1200": 2191}


def tile_clustering(dataloader, k, toy):
    kmeans = KMeans(n_clusters=k)
    res = []

    for i, (X, y) in enumerate(dataloader):
        vectorized = X[0].reshape((-1, 3))
        vectorized = np.float32(vectorized)
        cluster_labels = kmeans.fit_predict(vectorized)
        labels_count = Counter(cluster_labels)
        print(f"tile {i}: {labels_count}")
        res.append(kmeans.labels_)

        if toy:
            return res

    return res


def main(micron, filename, k):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=TILES_DATA / micron)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_classes", default=2)
    params = parser.parse_args()

    for set in SETS:
        write_tiles_wsi_tsv(micron, TILES_TSV / micron / f"scored_black_0.2_{set}.tsv")
        tsv = (
            TILES_TSV / f"{micron}/{filename}_{set}.tsv"
            if filename != ""
            else TILES_TSV / f"{micron}/{set}.tsv"
        )
        wsi_dataloaders = get_tiles_wsi_dataloaders(
            params, micron, tsv, transform=get_transfrom(resize_table[micron])
        )
        for wsi, dataloader in wsi_dataloaders:
            # tiles_names = pd.read_table(TILES_TSV / f"{micron}/{wsi}.tsv")["tile_name"]
            print(f"{wsi} clustering")
            tile_clustering(dataloader, k, toy=0)
            break


if __name__ == "__main__":
    main("400", "", 4)
