import pandas as pd
from config.constants import ROOT_DIR, WSI_TSV, TILES_TSV, SETS
from src.data.utils import write_file, delete_file_if_exist

#
# def make_featsee_tsv():
#     mirna_table = pd.read_table(ROOT_DIR / "tsv/mirna.tsv")
#     mirna_table = mirna_table[mirna_table["Stage"].isin(["I", "IV"])]
#     features_table = pd.read_table(ROOT_DIR / "tsv/400_selected_features.tsv")
#
#     result = pd.merge(
#         features_table,
#         mirna_table[
#             [
#                 "ID",
#                 "hsa-miR-6812-5p",
#                 "hsa-miR-133a-3p",
#                 "hsa-miR-320a-5p",
#                 "hsa-miR-3174",
#                 "hsa-miR-145-5p",
#                 "hsa-miR-1237-5p",
#                 "hsa-miR-1-3p",
#                 "hsa-miR-10396a-5p",
#                 "hsa-miR-3065-5p",
#                 "hsa-miR-762 18.29"
#             ]
#         ],
#         on=["ID"],
#         how="left",
#     )
#
#     result = result[~result.isnull().any(axis=1)]
#
#     # write features_list
#     fl_out_file = ROOT_DIR / "tsv/mirna_fl.tsv"
#     delete_file_if_exist(fl_out_file)
#     features_list = [
#         ["F0"], ["F1"], ["F2"], ["F3"], ["F4"], ["F5"], ["F6"], ["F7"], ["F8"], ["F9"],
#         ["m0"], ["m1"], ["m2"], ["m3"], ["m4"], ["m5"], ["m6"], ["m7"], ["m8"], ["m9"]
#     ]
#     write_file(features_list, fl_out_file, ["features"])
#
#     # write features
#     df_fl = pd.DataFrame(result)
#     out_file = ROOT_DIR / "tsv/mirna_dataset.tsv"
#     delete_file_if_exist(out_file)
#     df_fl.to_csv(out_file, index=False, sep="\t")


def make_featsee_tsv():
    for set in SETS:
        mirna_table = pd.read_table(ROOT_DIR / "tsv/mirna_top10.tsv").transpose()
        mirna_table.reset_index(inplace=True)
        mirna_table = mirna_table.rename(columns={'index':'ID'})

        features_table = pd.read_table(ROOT_DIR / f"tsv/400_selected_features_{set}.tsv")

        result = pd.merge(
            features_table,
            mirna_table[
                [
                    "ID",
                    "hsa-miR-6812-5p",
                    "hsa-miR-133a-3p",
                    "hsa-miR-320a-5p",
                    "hsa-miR-3174",
                    "hsa-miR-145-5p",
                    "hsa-miR-1237-5p",
                    "hsa-miR-1-3p",
                    "hsa-miR-10396a-5p",
                    "hsa-miR-3065-5p",
                    "hsa-miR-762"
                ]
            ],
            on="ID",
            how="left",
        )

        result = result[~result.isnull().any(axis=1)]

        # write features_list
        fl_out_file = ROOT_DIR / "tsv/mirna_fl.tsv"
        delete_file_if_exist(fl_out_file)
        features_list = [
            ["F0"], ["F1"], ["F2"], ["F3"], ["F4"], ["F5"], ["F6"], ["F7"], ["F8"], ["F9"],
            ["m0"], ["m1"], ["m2"], ["m3"], ["m4"], ["m5"], ["m6"], ["m7"], ["m8"], ["m9"]
        ]

        write_file(features_list, fl_out_file, ["features"])

        # write features
        df_fl = pd.DataFrame(result)
        out_file = ROOT_DIR / f"tsv/mirna_dataset_{set}.tsv"
        delete_file_if_exist(out_file)
        df_fl.to_csv(out_file, index=False, sep="\t")

#
# def make_wsi_features_tsv(micron, selected_features):
#     for set in SETS:
#         wsi_table = pd.read_table(WSI_TSV / "labeled.tsv")
#         wsi_list = wsi_table[wsi_table["Stage"].isin(["I", "IV"])][
#             "filename"
#         ].values.tolist()
#         rows = []
#         features_table = pd.read_table(
#             TILES_TSV / f"extracted_features/{micron}_features_{set}.tsv"
#         )
#
#         for wsi_name in wsi_list:
#             id = wsi_table[wsi_table["filename"] == wsi_name]["ID"].values[0]
#             id = f"VF{id[1:]}"
#             stage = wsi_table[wsi_table["filename"] == wsi_name]["Stage"].values[0]
#             features = [
#                 features_table[features_table["wsi_name"] == wsi_name][f].values[0]
#                 for f in selected_features
#             ]
#             rows.append([id, stage, wsi_name, *features])
#
#         out_path = ROOT_DIR / f"tsv/400_selected_features_{set}.tsv"
#         write_file(
#             rows, out_path, ["ID", "Stage", "wsi_name", "F0", "F1", "F2", "F3", "F4"]
#         )


def make_wsi_features_tsv(micron, selected_features):
    for set in SETS:
        wsi_table = pd.read_table(WSI_TSV / "labeled.tsv")
        wsi_list = wsi_table[
            "filename"
        ].values.tolist()
        rows = []
        features_table = pd.read_table(
            TILES_TSV / f"extracted_features/{micron}_features_{set}.tsv"
        )

        for wsi_name in wsi_list:
            id = wsi_table[wsi_table["filename"] == wsi_name]["ID"].values[0]
            id = f"VF{id[1:]}"

            try:
                stage = features_table[features_table["wsi_name"] == wsi_name]["stage"].values[0]
                features = [
                    features_table[features_table["wsi_name"] == wsi_name][f].values[0]
                    for f in selected_features
                ]
                rows.append([id, stage, wsi_name, *features])
            except:
                continue

        out_path = ROOT_DIR / f"tsv/400_selected_features_{set}.tsv"
        write_file(
            rows, out_path, ["ID", "Stage", "wsi_name",
                             "F0", "F1", "F2", "F3", "F4",
                             "F5", "F6", "F7", "F8", "F9"]
        )


def get_mirna_test(): 
    mirna = pd.read_table(ROOT_DIR / "tsv/mirna_dataset.tsv")
    test_set = pd.read_table(WSI_TSV / "test_set.tsv")    
    result = pd.merge(mirna, test_set, how="inner", on=["wsi_name"])
    result.to_csv(ROOT_DIR / 'tsv/mirna_test.tsv', sep="\t", index=False)


def main():
    #selected = ["F1520", "F3800", "F3972", "F3996", "F4237"]
    # selected = ["F138", "F142", "F143", "F144", "F145", "F154", "F380", "F397", "F399", "F423"]
    # make_wsi_features_tsv("400", selected_features=selected)
    make_featsee_tsv()
    #get_mirna_test()
    

if __name__ == "__main__":
    main()
