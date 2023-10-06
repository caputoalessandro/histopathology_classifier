import numpy as np
import plotly.express as px
from config.constants import *
from sklearn.preprocessing import normalize


def plot_confusion_matrix(micron, model, labels):
    data_path = Path(ASSETS / model / micron / "cf.npy")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "rb") as f:
        data = np.load(f)

    data_normed = normalize(data, axis=1, norm="l1")
    fig = px.imshow(
        data_normed,
        text_auto=True,
        labels=dict(x="Pred", y="Target", color="count"),
        x=labels,
        y=labels,
    )

    fig.write_html(ASSETS / model / micron / f"confusion_matrix.html")
    return fig


if __name__ == "__main__":
    plot_confusion_matrix("800", "lr=0.001_wd=0.0005")
