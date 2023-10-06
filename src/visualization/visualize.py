from functools import partial
from src.visualization.distributions import (
    plot_tiles_distributions,
    plot_wsi_distributions
)
from src.models.utils import FooParams
from src.visualization.utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

#MICRON
f = "400"
e = "800"
o = "1200"

# MODELS
tr = "lr=0.001_wd=0.0005"
LR_search = "LR_search"

# METRICS
recall = "recall"
accuracy = "accuracy"
loss = "loss"
specificity = "specificity"
precision = "precision"
balanced_accuracy = "balanced accuracy"
learning_rate = "learning rate"

# PLOTS
line = "line"
categories = "categories"
confusion_matrix = "confusion_matrix"
tiles_dist = "tiles_dist"
wsi_dist = "wsi_dist"

# SETS
training = "training"
validation = "validation"
test = "test"


def plot_sublines(fig, file, metrics, i):
    cols = px.colors.qualitative.Plotly
    df = pd.read_csv(file)

    for c, (name, m) in enumerate(metrics.items()):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[m],
                mode="lines",
                line=dict(width=2, color=cols[c]),
                name=name,
                showlegend=False if i !=1 else True),
            row=1,
            col=i
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=i)
    fig.update_yaxes(title_text="Accuracy", range=[0.3, 1], row=1, col=i)


def main_psl(micron, plot_filename):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training set", "Test set"))

    for i, (k, s) in enumerate([("tr", "training"), ("te", "test")]):
        metrics = {
            "p": {
                f"Denoised": f"400_denoised_davinci_1 - {s} balanced accuracy",
                f"Stain": f"400_stain_lambert - {s} balanced accuracy",
                f"CLAHE": f"400_cell_0.1_clahe_d1 - {s} balanced accuracy",
                f"Grayscale": f"400_cell_0.1_adam_gray_a0 - {s} balanced accuracy"
            },
            "frunf": {
                f"Freezed": f"400_cell_0.1_f2 - {s} balanced accuracy",
                f"Unfreezed": f"400_cell_0.1_adam - {s} balanced accuracy"
            },
            "f": {
                "1 block": f"400_cell_0.1_f1 - {s} balanced accuracy",
                "2 blocks": f"400_cell_0.1_f2 - {s} balanced accuracy",
                "3 blocks": f"400_cell_0.1_f3_d0 - {s} balanced accuracy",
                "4 blocks": f"400_cell_0.1_f4_a1 - {s} balanced accuracy"
            },
            "w": {
                f"Weighted": f"400_cell_0.1_weighted_a0 - {s} balanced accuracy",
                f"Unweighted": f"400_cell_0.1_adam - {s} balanced accuracy"
            },
            "opt": {
                f"sgd": f"400_cell_0.1 - {s} balanced accuracy",
                f"adam": f"400_cell_0.1_adam - {s} balanced accuracy"
            },
            "suns": {
                f"scored": f"400_cell_0.1 - {s} balanced accuracy",
                f"Unscored": f"400_filtered - {s} balanced accuracy"
            },
            "s": {
                "Cellular score 0.1": f"400_cell_0.1 - {s} balanced accuracy",
                "Cellular score 0.2": f"400_cell_0.2 - {s} balanced accuracy",
                "Nuclei score 0.1": f"400_nuclei_0.1 - {s} balanced accuracy",
                "Nuclei score 0.2": f"400_nuclei_0.2 - {s} balanced accuracy"
            },
            "funf": {
                f"Unfiltered": f"400_balanced - {s} balanced accuracy",
                f"Filtered": f"400_filtered - {s} balanced accuracy"
            },
            "bunb": {
                f"Balanced": f"400_balanced - {s} balanced accuracy",
                f"Unbalanced": f"400_unbalanced - {s} balanced accuracy"
            },
            "bb": {
                f"Baseline": f"400_unbalanced - {s} balanced accuracy",
                f"Best": f"400_cell_0.1_f2_adam_best - {s} balanced accuracy"
            },
            "m": {
                f"400 micron": f"400_unbalanced - {s} balanced accuracy",
                "800 micron": f"800_baseline_I-IV - {s} balanced accuracy",
                f"1200 micron": f"1200_baseline_I-IV - {s} balanced accuracy"
            },
            "lh": {
                "400 micron": f"400_baseline_lh - {s} balanced accuracy",
                "800 micron": f"800_baseline_lh - {s} balanced accuracy",
                "1200 micron": f"1200_baseline_lh - {s} balanced accuracy"
            },
            "4cls": {
                "400 micron": f"400_4class - {s} balanced accuracy"
            },
            "other": {
                "I vs IV": f"400_unbalanced - {s} balanced accuracy",
                "II vs III": f"400_II-III - {s} balanced accuracy",
                "I-II vs III-IV": f"400_lh - {s} balanced accuracy",
                "I vs II vs III vs IV": f"400_4class - {s} balanced accuracy"
            }
        }

        file = WANDB_TSV / f"{micron}_{k}_{plot_filename}.csv"
        plot_sublines(fig, file, metrics[plot_filename], i + 1)

        fig.update_layout(height=500, width=1200, font=dict(size=20))
        fig.update_annotations(font_size=22)
        fig.write_image(ASSETS / f"{plot_filename}.png", scale=1.5)


def plot_line_chart(params):
    fig = go.Figure()
    df = pd.read_csv(params.tsv_filename)

    for name, metric in params.metrics.items():
        fig.add_trace(go.Scatter(x=df.index, y=df[metric], mode="lines", name=name))

    fig.update_layout(
        title=params.plot_title, xaxis_title="Epoch", yaxis_title="Accuracy", title_x=0.5, showlegend=True
    )

    fig.update_xaxes(tick0=0, dtick=10)
    fig.update_yaxes(tick0=0, dtick=0.1, range=[0, 1])

    #fig.write_html(ASSETS / model / micron / f"{title}.html", auto_open=False)
    fig.write_image(ASSETS / f"{params.title}.png", scale=1.5)


def main():
    colors = {
        "green": "rgba(0, 204, 150,0.8)",
        "red": "rgba(239, 85, 59,0.8)",
        "blue": "rgba(99,110,250,0.8)",
        "purple": "rgba(171, 99, 250,0.8)"
    }

    colors_dict = {
        "400": colors["blue"],
        "800": colors["red"],
        "1200": colors["green"]
    }

    params = FooParams(
        function=tiles_dist,
        micron=f,
        mode="",
        #tsv_filename=WANDB_TSV / "400_te_opt.csv",
        tsv_filename="I-IV",
        color=colors["blue"],
        title=f"unbal_blue",
        plot_title=f"400 micron",
        max=10000,
    )

    params.title = params.mode if params.title == None else params.title
    get_visualize_args(params)

    functions = {
        # categories: plot_categories,
        # confusion_matrix: partial(plot_confusion_matrix, micron, model, title),
        line: partial(plot_line_chart, params),
        tiles_dist: partial(plot_tiles_distributions, params),
        wsi_dist: partial(plot_wsi_distributions, params)
    }

    functions[params.function]()


if __name__ == "__main__":
    #main()
    main_psl("400", "other")
