import torch
import argparse
import params as pp
import wandb
import os
from config.keys import WANDB_API_KEY
from src.visualization.distributions import get_tiles_dataset_distribution
import re
import random
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import csv
import numpy as np
from config.constants import *


from torchmetrics import (
    MetricCollection,
    Accuracy,
    Recall,
    Specificity,
    Precision,
    ConfusionMatrix,
)


def save_stats(model_name, micron, set_name, result: dict, epoch):
    model_hyper = model_name.replace("_resnet18", "")
    Path(ROOT_DIR / "assets" / model_hyper / micron).mkdir(parents=True, exist_ok=True)
    out_path = ROOT_DIR / "assets" / model_hyper / micron / (set_name + "_metrics.tsv")

    if epoch == 0:
        delete_file_if_exist(out_path)

    with open(out_path, "a") as f:
        tsv_writer = csv.writer(f, delimiter="\t")

        if epoch == 0:
            row = [
                "accuracy",
                "recall",
                "specificity",
                "precision",
                "balanced accuracy",
                "learning rate",
                "loss",
            ]
            tsv_writer.writerow(row)

        tsv_writer.writerow(result.values())


def get_stats(preds, target, loss, params):
    params.task = "Binary" if params.num_classes == 2 else "Multiclass"

    metric_collection = MetricCollection(
        [
            Accuracy(params.task.lower(), num_classes=params.num_classes).to(
                params.device
            ),
            Recall(params.task.lower(), num_classes=params.num_classes).to(
                params.device
            ),
            Specificity(params.task.lower(), num_classes=params.num_classes).to(
                params.device
            ),
            Precision(params.task.lower(), num_classes=params.num_classes).to(
                params.device
            ),
        ]
    )

    result = metric_collection(preds, target)
    balanced_accuracy = (
        result[f"{params.task}Recall"] + result[f"{params.task}Specificity"]
    ) / 2
    result[f"Balanced accuracy"] = balanced_accuracy
    result = {k: v.item() for k, v in result.items()}
    result["Learning rate"] = params.lr
    result["Loss"] = loss
    return result


def save_confusion_matrix(preds, target, params):
    confmat = ConfusionMatrix(
        task=params.task.lower(), num_classes=params.num_classes
    ).to(params.device)

    res = confmat(preds, target)
    res = res.cpu().numpy()
    file = ASSETS / params.run_name / params.micron / "cf.npy"
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb+") as f:
        np.save(f, res)


def get_dict_stats(data_list):
    final_result = {}
    for set, data in data_list:
        keys = [
            f"{set} accuracy",
            f"{set} recall",
            f"{set} specificity",
            f"{set} precision",
            f"{set} balanced accuracy",
            f"{set} learning rate",
            f"{set} loss",
        ]

        res = {key: val for key, val in zip(keys, data.values())}
        final_result = {**final_result, **res}
    return final_result


def freeze_resnet18(key, model):
    if key == "none":
        return

    freeze_map = {
        "all": slice(None, None),
        "1res_b": slice(0, 5),
        "2res_b": slice(0, 6),
        "3res_b": slice(0, 7),
        "4res_b": slice(0, 8),
    }

    childs = list(model.children())

    for child in childs[freeze_map[key]]:
        print(f"{child} freezing")
        for param in child.parameters():
            param.requires_grad = False


def build_resnet18(device, pretrained, num_class, freeze_key="none"):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights if pretrained else None)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
    #                        bias=False)
    model.fc = nn.Linear(512, num_class)
    # freeze_resnet18(freeze_key, model)
    model.to(device)
    return model


def build_cifar_resnet(device, pretrained, num_class, freeze_key="none"):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights if pretrained else None, num_classes=num_class)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    freeze_resnet18(freeze_key, model)
    model.to(device)
    print(f"The model is running on: {next(model.parameters()).device}")
    return model


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_validation_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def wandb_init(params):
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    if params.wandb_flag:
        wandb.init(
            name=params.run_name,
            dir=ROOT_DIR,
            # set the wandb project where this run will be logged
            project="histopatho-cancer-grading",
            # track hyperparameters and run metadata
            config={
                "learning_rate": params.lr,
                "weight decay": params.wd,
                "batch_size": params.batch_size,
                "tiles micron": params.micron,
                "classes": params.num_classes,
                "lr_search": params.lr_search,
                "architecture": "RESNET18",
                "epochs": params.epochs,
            },
        )


class FooParams(pp.Params):
    tsv = {}
    task = None
    distributions = {}
    function = None
    datasets = None
    micron = None
    lr_search = None
    epochs = None
    lr = None
    wd = None
    momentum = None
    batch_size = None
    toy_flag = 0
    run_name = None
    freeze_key = None
    tiles_type = None
    gpu = None
    training_set = None
    validation_set = None
    test_set = None
    model = None
    model_name = None
    model_path = None
    loss_fn = None
    optimizer = None
    task = None
    device = None
    wandb_flag = 1
    num_classes = None
    cifar = None
    cuda_on = 1
    preprocessing = None
    memory_factor = None
    subset = None
    title = None
    data_path = None
    sets = None
    accumulation = None
    loss_weights = None
    labels = None
    tsv_filename = None
    filename = None
    loss_flag = False
    folder_data = None
    make_tsv = None
    make_fe = None
    mode = None
    we = None
    opt = None


class TensorCat:
    def __init__(self):
        self.total_preds = None
        self.total_targets = None

    def __call__(self, pred, target):
        try:
            self.total_preds = torch.cat((self.total_preds, torch.argmax(pred, 1)), 0)
            self.total_targets = torch.cat(
                (self.total_targets, target.clone().detach()), 0
            )
            # self.total_targets.append(target)
        except:
            self.total_preds = torch.argmax(pred, 1)
            # self.total_targets = torch.argmax(target, 1)
            self.total_targets = target.clone().detach()


class SaveBestModel:
    def __init__(self):
        self.max_acc = 0

    def __call__(self, acc, params, epoch):
        bound = -1 if params.toy_flag else 30
        if (acc >self.max_acc) and epoch > bound:
            self.max_acc = acc
            print("save model")
            params.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(params.model.state_dict(), params.model_path)


def delete_file_if_exist(file):
    if file.is_file():
        file.unlink()
        print(f"{file} removed")


def get_avg_loss(loss_sum, dataloader, params):
    mean_factor = len(dataloader) / params.accumulation
    return loss_sum / mean_factor


def early_stopping_check(early_stopping, model, model_hyper, test_loss, epoch):
    early_stopping(test_loss)

    if early_stopping.save_model:
        print("save model")
        torch.save(
            model.state_dict(), SAVED_MODELS_DIR / (model_hyper + "_resnet18.pth")
        )

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        return True


def get_experiment_args(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fd", "--folder_data", nargs="?", default=str(params.folder_data), type=str
    )
    parser.add_argument(
        "-m", "--micron", nargs="?", default=str(params.micron), type=str
    )
    parser.add_argument(
        "-pp", "--preprocessing", nargs="?", default=params.preprocessing, type=int
    )
    parser.add_argument(
        "-lrs", "--lr_search", nargs="?", default=params.lr_search, type=int
    )
    parser.add_argument("-e", "--epochs", nargs="?", default=params.epochs, type=int)
    parser.add_argument("-lr", nargs="?", default=params.lr, type=float)
    parser.add_argument("-wd", nargs="?", default=params.wd, type=float)
    parser.add_argument(
        "-bs", "--batch_size", nargs="?", default=params.batch_size, type=int
    )
    parser.add_argument(
        "-tf", "--toy_flag", nargs="?", default=params.toy_flag, type=int
    )
    parser.add_argument(
        "-wdb", "--wandb_flag", nargs="?", default=params.wandb_flag, type=int
    )
    parser.add_argument("-gpu", nargs="?", default=params.wandb_flag, type=int)
    parser.add_argument(
        "-rn", "--run_name", nargs="?", default=params.run_name, type=str
    )
    parser.add_argument(
        "-fk", "--freeze_key", nargs="?", default=params.freeze_key, type=str
    )
    parser.add_argument(
        "-mn", "--model_name", nargs="?", default=params.model_name, type=str
    )
    parser.add_argument(
        "-mm", "--momentum", nargs="?", default=params.momentum, type=float
    )
    parser.add_argument(
        "-nc", "--num_classes", nargs="?", default=params.num_classes, type=int
    )
    parser.add_argument("-co", "--cuda_on", nargs="?", default=params.cuda_on, type=int)
    parser.add_argument(
        "-tsv", "--tsv_filename", nargs="?", default=params.tsv_filename, type=str
    )
    parser.add_argument("-w", "--we", nargs="?", default=params.we, type=int)
    parser.add_argument("-o", "--opt", nargs="?", default=params.opt, type=str)
    params.update(vars(parser.parse_args()))
    return parser.parse_args()


def set_loss_weights(params):
    t_dist = get_tiles_dataset_distribution(params, "training_set")
    num_samples = sum(list(t_dist.values()))
    weights = [num_samples / val for val in t_dist.values()]
    norm_w = [n / sum(weights) for n in weights]
    print(num_samples)
    print(weights)
    print(norm_w)
    params.loss_weights = torch.Tensor(norm_w).to(params.device)
    params.loss_flag = True


def get_labels(params):
    labels_table = {
        "2_None": ["I-II", "III-IV"],
        "4_None": ["I", "II", "III", "IV"],
    }
    subset_labels = re.findall("\w+", params.subset) if params.subset else None
    return labels_table.get(f"{params.num_classes}_{params.subset}", subset_labels)
