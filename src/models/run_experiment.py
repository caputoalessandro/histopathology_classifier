from utils import FooParams, wandb_init
from src.models.predict import predict_model
import torch
from utils import (
    get_experiment_args,
    set_seed,
    build_resnet18,
    set_loss_weights,
    freeze_resnet18
)
from src.data.build_dataloader import get_tiles_dataloaders
from torchvision import transforms
from src.models.training import train_model
from src.visualization.distributions import wandb_distributions
from config.constants import TILES_DATA


FREEZE_1RESB = "1res_b"
FREEZE_2RESB = "2res_b"
FREEZE_3RESB = "3res_b"
FREEZE_4RESB = "4res_b"


def get_training_transfroms(size):
    return transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(size, size), padding=4),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def get_test_transfroms(size):
    return transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


resize_table = {"50": 100, "100": 365, "400": 1460, "800": 1800, "1200": 2191}
acc_table = {"50": 8, "100": 2, "400": 8, "800": 8, "1200": 16}
batch_table = {"50": 8, "100": 16, "400": 4, "800": 4, "1200": 2}


def print_hyperparms(params, resize, seed, train_transf, test_transf):
    print(
        "\n############################# DATASET #####################################################"
    )
    print(f"micron: {params.micron}")
    print(f"image resize = {resize}")
    print(f"training set transformations: {train_transf}")
    print(f"test set transformations: {test_transf}")
    print(f"tiles dir: {params.data_path}")
    print(f"tsv file: {params.tsv_filename}")
    print(f"training set size: {len(params.training_set) * params.batch_size}")
    print(f"test set size: {len(params.test_set) * params.batch_size}")
    print(
        "############################# NET PARAMETERS ##############################################"
    )
    print(f"optimizer = {params.optimizer}")
    print(f"lr = {params.lr} | search: {bool(params.lr_search)}")
    print(f"loss: {params.loss_fn} | weighted {params.loss_flag}")
    print(f"wd = {params.wd}")
    print(f"momentum = {params.momentum}")
    print(f"batch size = {params.batch_size} | accumulation = {params.accumulation}")
    print(f"Random seed set as {seed}")
    print(f"The model is running on: {next(params.model.parameters()).device}")
    print(
        "###########################################################################################\n"
    )


def main():
    # PARAMETERS
    params = FooParams(
        labels=["I-II","III-IV"],
        #labels=["I","IV"],
    )
    print(f"LABELS: {params.labels}")
    get_experiment_args(params)
    params.accumulation = acc_table[params.micron]
    resize = resize_table[params.micron]
    params.batch_size = batch_table[params.micron]
    wandb_init(params)
    seed = 42
    set_seed(seed)
    params.data_path = TILES_DATA / f"{params.folder_data}"

    # DATASET
    train_transf = get_training_transfroms(resize)
    test_transf = get_test_transfroms(resize)
    params.training_set = get_tiles_dataloaders(params, train_transf, ["training_set"])
    params.test_set = get_tiles_dataloaders(params, test_transf, ["test_set"])
    
    # MODEL
    params.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and params.cuda_on else "cpu"
    )
    
    if params.we==1:
        set_loss_weights(params)
        params.loss_fn = torch.nn.CrossEntropyLoss(weight=params.loss_weights)
    else:
        params.loss_fn = torch.nn.CrossEntropyLoss()
    
    params.model = build_resnet18(
        num_class=params.num_classes, device=params.device, pretrained=False
    )
    # freeze_resnet18("2res_b", params.model)
    
    if params.opt=="sgd":
        params.optimizer = torch.optim.SGD(
            params.model.parameters(),
            lr=params.lr,
            weight_decay=params.wd,
        momentum=params.momentum,
        )
    elif params.opt=="adam":
        params.optimizer = torch.optim.Adam(
            params=params.model.parameters(),
            lr=params.lr,
            weight_decay=params.wd
        )

    print_hyperparms(params, resize, seed, train_transf, test_transf)

    # TRAINING
    wandb_distributions(params)
    train_model(params)

    # TEST
    params.model.load_state_dict(
        torch.load(params.model_path, map_location=params.device)
    )
    predict_model(params)


if __name__ == "__main__":
    main()

    # set_loss_weights(params)
    # params.loss_fn = torch.nn.CrossEntropyLoss(weight=params.loss_weights)
    # params.optimizer = torch.optim.Adam(params=params.model.parameters(), lr=params.lr, weight_decay=params.wd)

    # params.validation_set = get_tiles_dataloaders(
    #     params, test_transf, ["validation_set"]
    # )

    # params.model_path = SAVED_MODELS_DIR / (params.run_name + ".pt")
    # params.model.load_state_dict(
    #     torch.load(params.model_path, map_location=params.device)["model"]
    # )
