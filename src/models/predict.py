import torch
import wandb
from src.models.utils import (
    TensorCat,
    get_avg_loss,
)
from src.models.utils import (
    get_stats,
    save_confusion_matrix,
    get_dict_stats,
)
from src.visualization.confusion_matrix import plot_confusion_matrix
from torchvision import transforms


def get_test_transfroms():
    return transforms.Compose(
        [
            # transforms.Resize(size=(1800, 1800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def save_features(params, input):
    return_nodes = {"layer4.1.relu_1": "layer4"}
    inter_model = create_feature_extractor(params.model, return_nodes=return_nodes)
    features = inter_model(input)


def predict_model(params):
    params.model.eval()
    tensor_cat = TensorCat()
    loss_sum = 0
    use_amp = True if params.device == "cuda:0" else False

    with torch.no_grad():
        for X, y in params.test_set:
            X, y = X.to(params.device), y.to(params.device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                pred = params.model(X)
                loss = params.loss_fn(pred, y)

            loss_sum += loss.item()
            tensor_cat(pred, y)

            if params.toy_flag:
                break

    save_confusion_matrix(tensor_cat.total_preds, tensor_cat.total_targets, params)

    stats = get_stats(
        tensor_cat.total_preds,
        tensor_cat.total_targets,
        get_avg_loss(loss_sum, params.test_set, params),
        params,
    )

    # save_stats(params.run_name, params.micron, "test", stats, 0)
    wandb_data = [("best_model_test", stats)]
    dict_stats = get_dict_stats(wandb_data)
    cm = plot_confusion_matrix(params.micron, params.run_name, params.labels)

    if params.wandb_flag:
        wandb.log({"Confusion matrix": cm})
        wandb.log(dict_stats)
