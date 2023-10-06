import pandas as pd
from plotly.subplots import make_subplots
import math
from src.visualization.utils import *
from config.constants import *
import wandb
import plotly.express as px


# def get_nuclei_score_distributions(micron, scores, make_tsv):
#     pipe_args = [("cell", str(score), "scored_black_0.2") for score in scores]
#     if make_tsv:
#         make_scored_tsv(micron=micron, pipe_args=pipe_args)
#     result = []
#
#     for score in scores:
#         train_table = pd.read_table(TILES_TSV / micron / f"scored_nuclei_{score}_scored_black_0.2_training_set.tsv")
#         test_table = pd.read_table(TILES_TSV / micron / f"scored_nuclei_{score}_scored_black_0.2_test_set.tsv")
#         # whole_table = pd.concat([train_table, test_table], axis=0)
#         data = [len(train_table[train_table["tile_label"] == stage]) for stage in STAGES]
#         result.append(data)
#
#     return pd.DataFrame(columns=STAGES, index=scores, data=result)


# def plot_nuclei_score_distributions(micron, scores, make_tsv):
#     df = get_nuclei_score_distributions(micron, scores, make_tsv)
#     print()


def plot_wsi_distributions(params, file=WSI_TSV / "labeled.tsv"):
    dfs = get_split_stats(file, params)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Whole dataset - {dfs['whole_dataset'].sum().sum()} images",
            f"Training set - {dfs['training_set'].sum().sum()} images",
            #f"Validation set - {dfs['validation_set'].sum().sum()} images",
            f"Test set - {dfs['test_set'].sum().sum()} images",
        ),
    )

    fig.add_bar(y=dfs["whole_dataset"], row=1, col=1, text=dfs["whole_dataset"])
    fig.add_bar(y=dfs["training_set"], row=1, col=2, text=dfs["training_set"])
    #fig.add_bar(y=dfs["validation_set"], row=1, col=3, text=dfs["validation_set"])
    fig.add_bar(y=dfs["test_set"], row=1, col=3, text=dfs["test_set"])

    max = dfs["whole_dataset"].to_numpy().max()

    if params.mode == "I_IV":
        labels = ["I", "IV"]
    elif params.mode == "lh":
        labels = ["I - II", "III - IV"]
    elif params.mode == "all":
        labels = ["I", "II", "III", "IV"]

    fig.update_xaxes(
        ticktext=labels,
        tickvals=[0, 1, 2, 3] if params.mode == "all" else [0, 1]
    )

    fig.update_yaxes(
        title_font_size=10,
        title_standoff=0,
        title_text="Count",
        range=[0, max + math.ceil(max * 0.10)],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        # title_text=f"WSI distributions",
    )
    Path(ASSETS).mkdir(parents=True, exist_ok=True)
    fig.write_image(ASSETS / f"WSI_distribution_{params.title}.png",  width=1000, height=600)


def get_tiles_dataset_distribution(params, set):
    tsv = f"{TILES_TSV / params.micron / set}.tsv" if params.tsv_filename == "" else TILES_TSV / f"{params.micron}/{params.tsv_filename}_{set}.tsv"
    table = pd.read_table(tsv)
    print(tsv)
    stage_list = table["tile_label"].to_list()
    count = dict(Counter(stage_list))

    if params.mode == "lh":
        count = {"I - II": count["I"] + count["II"], "III - IV": count["III"] + count["IV"]}
    elif params.mode == "I_IV":
        count = {"I": count["I"], "IV": count["IV"]}

    return count
    

def get_tiles_datasets_distributions(params):
    return {
        set: get_tiles_dataset_distribution(params, set)
        for set in SETS
    }


def wandb_distributions(params):
    distributions = get_tiles_datasets_distributions(params)
    if params.wandb_flag:
        for s, dict_values in distributions.items():
            data = [
                [label, val]
                for (label, val) in zip(dict_values.keys(), dict_values.values())
            ]
            table = wandb.Table(data=data, columns=["label", "value"])
            r = {
                f"{s}_{params.run_name}_distribution": wandb.plot.bar(
                    table,
                    "label",
                    "value",
                    title=f"{s} {params.micron} - {sum(distributions[s].values())}",
                )
            }
            params.distributions.update(r)


def plot_tiles_distributions(params):
    distributions = get_tiles_datasets_distributions(params)
    distributions = pd.DataFrame.from_dict(distributions)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=([f"{set.replace('_',' ')}  {distributions[set].sum()} tiles" for set in SETS]),
        horizontal_spacing=0.2
    )

    for col, set in enumerate(SETS):
        fig.add_bar(
            y=distributions[set],
            x=distributions.index,
            row=1,
            col=col + 1,
            text=distributions[set],

        )


    max = distributions.to_numpy().max()

    fig.update_yaxes(
        title_font_size=20,
        title_standoff=0,
        title_text="Count",
        range=[0, params.max] if max else [0, max + math.ceil(max * 0.10)],
    )

    fig.update_traces(textposition="outside", marker_color=params.color)

    fig.update_layout(
        title=params.plot_title,
        title_x=0.5,
        width=500,
        height=500,
        showlegend=False,
        font=dict(size=20)
    )
    # fig.for_each_yaxis(lambda axis: axis.title.update(font=dict(size=18)))
    fig.update_annotations(font_size=24)

    # fig.write_html(
    #     ASSETS / f"{params.micron}micron_{params.title}.html", auto_open=True
    # )

    fig.write_image(f"{ASSETS / params.micron}_{params.title}.png", width=800, height=600)

    title = f"Dataset distribution ({params.micron}\u03BC, {distributions.sum().sum()} tiles)"
    return fig, title

