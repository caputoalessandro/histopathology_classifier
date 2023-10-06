from plotly.subplots import make_subplots
import math
from src.visualization.utils import *
import plotly.graph_objects as go
from functools import partial
from config.constants import *


def get_categories_functions(df):
    return {
        "Age": partial(
            get_stats,
            df,
            [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)],
            "Age",
            range_query,
        ),
        "BMI": partial(
            get_stats,
            df,
            [
                (0, 16),
                (16, 18.5),
                (18.5, 25),
                (25, 30),
                (30, 35),
                (35, 40),
                (40, 45),
            ],
            "BMI",
            range_query,
        ),
        "Sex": partial(get_stats, df, ["Male", "Female"], "Sex", categorical_query),
        "Smoking": partial(
            get_stats, df, ["Smoker", "Never", "Former"], "Smoking", categorical_query
        ),
    }


def categorical_query(df, values, feature, stage):
    columns = []
    row_val = []

    for value in values:
        columns.append(f"{value}")
        table = df[
            (df[feature] == value)
            & ((df["Stage"] == stage[0]) | (df["Stage"] == stage[1]))
        ]
        value = len(table)
        row_val.append(value)

    return columns, row_val


def range_query(df, ranges, feature, stage):
    columns = []
    row_val = []

    for start, stop in ranges:
        columns.append(f"{start}-{stop}")
        table = df[
            df[feature].between(start, stop, inclusive="right")
            & ((df["Stage"] == stage[0]) | (df["Stage"] == stage[1]))
        ]
        value = len(table)
        row_val.append(value)

    return columns, row_val


def get_stats(df, query_value, query_feature, query):
    stages = [("I", "II"), ("III", "IV")]
    rows_val = []
    row_label = []
    columns = []

    for stage in stages:
        row_label.append(f"{stage[0]}-{stage[1]}")
        columns, values = query(df, query_value, query_feature, stage)
        rows_val.append(values)

    return pd.DataFrame(rows_val, index=row_label, columns=columns)


def plot_categories(data_path=WSI_TSV / "labeled.tsv"):
    df = pd.read_table(data_path)
    functions = get_categories_functions(df)
    stages = ["I-II", "III-IV"]
    titles = [f"{k} ({stage})" for k, v in functions.items() for stage in stages]

    fig = make_subplots(
        rows=len(functions.keys()), cols=len(stages), subplot_titles=titles
    )

    for row, category in enumerate(functions.keys()):
        res = functions[category]()
        max = res.to_numpy().max()

        for column, stage in enumerate(stages):
            fig.update_yaxes(
                title_font_size=10,
                title_standoff=0,
                title_text="Count",
                range=[0, max + math.ceil(max * 0.10)],
                row=row + 1,
                col=column + 1,
            )

            fig.add_trace(
                go.Bar(
                    x=res.columns,
                    y=res.loc[stage],
                    name=f"{stage}",
                    text=res.loc[stage],
                    width=0.5,
                ),
                row=row + 1,
                col=column + 1,
            )

    fig.update_layout(showlegend=False, autosize=False, width=1000, height=2000)

    fig.update_traces(textposition="outside")
    fig.write_image(ASSETS / f"WSI_categories_distribution.pdf")
    fig.write_html(ASSETS / f"WSI_categories_distribution.html", auto_open=True)
