from config.constants import (
    SAVED_MODELS_DIR,
    ROOT_DIR,
    TILES_TSV,
    WSI_TSV,
    TILES_DATA,
    SETS,
)
from src.models.utils import build_resnet18
import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from src.data.build_dataloader import set_dataset_paths
from torchvision import transforms
from utils import FooParams
import numpy as np
import pandas as pd
from src.data.build_dataloader import get_tile_dataloader
from src.data.utils import write_file, delete_folder_if_exist, delete_file_if_exist
import itertools

resize_table = {"50": 300, "400": 1460, "800": 1800, "1200": 2191}


# def get_wsi_features(params, dataloader, feature_extractor, layer, wsi_name, stage):
#     # feature_extractor.model.eval()
#     FEATURES = []
#     use_amp = True if params.device == "cuda:0" else False

#     with torch.no_grad():
#         for batch, (X, y) in enumerate(dataloader):
#             X = X.to(params.device)

#             with torch.autocast(
#                 device_type="cuda", dtype=torch.float16, enabled=use_amp
#             ):
#                 pred = feature_extractor(X)
#                 FEATURES.append([wsi_name,stage,*pred[layer].tolist()[0]])

#             if params.toy_flag and batch == 4:
#                  FEATURES.append([wsi_name,stage,*pred[layer].tolist()[0]])

#     return FEATURES


# def feature_extraction(params, set, dataloaders, feature_extractor, layer):
#     print(f"{set} FEATURE EXTRACION")
#     ROWS = []

#     for i, (wsi_name, stage, dataloader) in enumerate(dataloaders):
#         wsi_feature = get_wsi_features(params, dataloader, feature_extractor, layer, wsi_name,stage)
#         ROWS = ROWS + wsi_feature
#         print(f"{i + 1} of {len(dataloaders)} completed | {wsi_name} feature len = {len(wsi_feature[0])-2}")
#         if params.toy_flag and i == 1:
#             break

#     # write feature list file
#     columns = [f"F{i}" for i in range(len(ROWS[0])-2)]
#     df_fl = pd.DataFrame(columns)
#     df_fl.columns = ["features"]
#     fl_out_file = ROOT_DIR / f"data/extracted_features/{params.micron}_{set}_fl.tsv"
#     delete_file_if_exist(fl_out_file)
#     df_fl.to_csv(fl_out_file, mode='a', index=False)

#     # write extracted features file
#     columns.insert(0, "stage")
#     columns.insert(0, "wsi_name")
#     f_out_file = ROOT_DIR / f"data/extracted_features/{params.micron}_{set}_features.tsv"
#     delete_file_if_exist(f_out_file)
#     write_file(wsi_feature, f_out_file, columns)

#     print("FINISH")


def get_wsi_features(params, dataloader, feature_extractor, layer):
    # feature_extractor.model.eval()
    use_amp = True if params.device == "cuda:0" else False

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(params.device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                pred = feature_extractor(X)
                try:
                    FEATURES = torch.cat((pred[layer], FEATURES), 0)
                    #print(f"concat tensor size {FEATURES.size()}")
                except:
                    FEATURES = pred[layer]

            if params.toy_flag and batch == 4:
                return FEATURES.tolist()

    return FEATURES.tolist()


def feature_extraction(params, set, dataloaders, feature_extractor, layer):
    print(f"{set} FEATURE EXTRACION")
    ROWS = []

    for i, (wsi_name, stage, dataloader) in enumerate(dataloaders):
        wsi_feature = get_wsi_features(params, dataloader, feature_extractor, layer)
        stage = "I-II" if stage == "I" or stage == "II" else "III-IV"
        to_append = [[wsi_name, stage, *f] for f in wsi_feature]

        for f in to_append:
            ROWS.append(f)

        # merged = list(itertools.chain.from_iterable(wsi_feature))
        # merged.insert(0, stage)
        # merged.insert(0, wsi_name)
        # ROWS.append(merged)

        print(
            f"{i + 1} of {len(dataloaders)} completed | {wsi_name}"
        )
        if params.toy_flag and i == 1:
            break

    return ROWS
    print("FINISH")


def get_test_transfroms(size):
    return transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def get_tiles_wsi_dataloaders(params, micron, tsv, transform=None):
    table = pd.read_table(tsv)
    res = []
    for wsi_name in table["wsi_name"].unique().tolist():
        try:
            to_load = TILES_TSV / micron / f"{wsi_name}.tsv"
            dataloader = get_tile_dataloader(
                to_load,
                params.data_path,
                params.batch_size,
                params.num_classes,
                transform,
            )
            stage = table[table["wsi_name"] == wsi_name].iloc[0]["tile_label"]
            res.append((wsi_name, stage, dataloader))
            # print(f"{wsi_name} tiles loaded")

        except Exception as e:
            print(str(e))

    return res


def save_model_features(params):
    # MODEL
    params.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and params.cuda_on else "cpu"
    )
    model = build_resnet18(
        num_class=params.num_classes, device=params.device, pretrained=False
    )
    model.load_state_dict(torch.load(params.model_path, map_location=params.device))
    # t, v = get_graph_node_names(model)

    # FEATURE EXTRACTOR
    layer = "flatten"
    return_nodes = {layer: layer}
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    # train_nodes, eval_nodes = get_graph_node_names(model)

    # DATASET
    FEATURES = []
    for set in SETS:
        set_dataset_paths(params, [set], params.filename)
        params.data_path = TILES_DATA / params.micron

        tsv = (
            TILES_TSV
            / params.micron
            / (f"{params.filename}_{set}.tsv" if params.filename else f"{set}.tsv")
        )

        test_transf = get_test_transfroms(resize_table[params.micron])
        dataloaders = get_tiles_wsi_dataloaders(params, params.micron, tsv, test_transf)

        # EXTRACTION
        FEATURES = feature_extraction(params, set, dataloaders, feature_extractor, layer)

        # write feature list file
        columns = [f"F{i}" for i in range(len(FEATURES[0]) - 2)]
        df_fl = pd.DataFrame(columns)
        df_fl.columns = ["features"]
        fl_out_file = TILES_TSV / f"extracted_features/{params.micron}_fl.tsv"
        delete_file_if_exist(fl_out_file)
        df_fl.to_csv(fl_out_file, mode="a", index=False)

        # write extracted features file
        columns.insert(0, "stage")
        columns.insert(0, "wsi_name")
        f_out_file = TILES_TSV / f"extracted_features/{params.micron}_features_{set}.tsv"
        delete_file_if_exist(f_out_file)
        write_file(FEATURES, f_out_file, columns)


# def return_best_k_scored_tile(wsi_name, set, micron, k):
#     table = pd.read_table(TILES_TSV / micron / f"scored_nuclei_{set}.tsv")
#     to_take = k*2 if table[table["wsi_name"] == wsi_name].iloc[0]["tile_label"] == "IV" else k
#     tile_table = table[table["wsi_name"] == wsi_name]
#     return tile_table.sort_values(by=["score"]).tail(to_take).values.tolist()


def return_best_k_scored_tile(wsi_name, micron, k):
    table = pd.read_table(TILES_TSV / micron / f"scored_cell.tsv")
    tile_table = table[table["wsi_name"] == wsi_name]
    print(wsi_name)
    return tile_table.sort_values(by=["score"]).tail(k).values.tolist()


def write_tiles_wsi_tsv(micron, tsv, set, k, toy):
    tiles_table = pd.read_table(tsv)
    print(f"{set} best tile extraction")

    for i, wsi_name in enumerate(tiles_table["wsi_name"].unique().tolist()):
        out_path = TILES_TSV / micron / f"{wsi_name}.tsv"
        delete_file_if_exist(out_path)
        best_tiles = return_best_k_scored_tile(wsi_name, micron, k)
        # print(f"gest tiles lenght ={len(best_tiles)}")
        write_file(
            best_tiles, out_path, ["wsi_name", "tile_name", "tile_label", "score"]
        )
        print(f"{i + 1} executed")

        if toy:
            break


def make_tiles(params, k):
    for set in SETS:
        tsv = (
            TILES_TSV
            / params.micron
            / (f"{params.filename}_{set}.tsv" if params.filename else f"{set}.tsv")
        )
        write_tiles_wsi_tsv(params.micron, tsv, set, k, params.toy_flag)


def features_size_test(micron):
    table = pd.read_table(ROOT_DIR / f"data/extracted_features/{micron}_features.tsv")
    print(f"Nan values table: {table.isnull().values.any()}")


def main():
    params = FooParams(
        cuda_on=0,
        micron="400",
        batch_size=2,
        num_classes=2,
        toy_flag=0,
        filename="scored_black_0.2",
        make_tsv=1,
        make_fe=1,
    )
    params.model_path = SAVED_MODELS_DIR / "400_7499.pt"

    if params.make_tsv:
        make_tiles(params, 1)
        features_size_test(params.micron)

    if params.make_fe:
        save_model_features(params)


if __name__ == "__main__":
    main()
