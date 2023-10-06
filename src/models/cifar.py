from utils import FooParams, build_cifar_resnet, get_experiment_args, wandb_init
from src.models.training import train_model
from src.models.predict import predict_model
import torchvision.transforms as transforms
import torch
import torchvision
from config.constants import ROOT_DIR
from config.constants import SAVED_MODELS_DIR
from datetime import datetime


# train_transform = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(size=(32, 4), scale=(0.8, 1)),
#         transforms.ToTensor(),
#         normalize,
#     ]
# )

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]
)

val_trasnform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)


CIFARTRAIN = {
    "100": torchvision.datasets.CIFAR100(
        root=ROOT_DIR / "data/other",
        train=True,
        download=True,
        transform=train_transform,
    ),
    "10": torchvision.datasets.CIFAR10(
        root=ROOT_DIR / "data/other",
        train=True,
        download=True,
        transform=train_transform,
    ),
}

CIFARTEST = {
    "100": torchvision.datasets.CIFAR100(
        root=ROOT_DIR / "data/other",
        train=False,
        download=True,
        transform=val_trasnform,
    ),
    "10": torchvision.datasets.CIFAR10(
        root=ROOT_DIR / "data/other",
        train=False,
        download=True,
        transform=val_trasnform,
    ),
}


def main():
    params = FooParams(
        cuda_on=0,
        epochs=1,
        lr=1e-1,
        wd=1e-4,
        momentum=0.9,
        batch_size=2,
        task="Multiclass",
        wandb_flag=0,
        freeze_key="none",
        toy_flag=1,
        num_classes=10,
        model_name="cifar.pth",
        run_name="cifar",
    )
    get_experiment_args(params)

    params.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and params.cuda_on else "cpu"
    )

    params.loss_fn = torch.nn.CrossEntropyLoss()
    params.model = build_cifar_resnet(
        num_class=params.num_classes,
        device=params.device,
        pretrained=False,
        freeze_key=params.freeze_key,
    )
    params.optimizer = torch.optim.SGD(
        params.model.parameters(),
        lr=params.lr,
        weight_decay=params.wd,
        momentum=params.momentum,
    )

    params.training_set = torch.utils.data.DataLoader(
        CIFARTRAIN[str(params.num_classes)],
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=8,
    )

    params.validation_set = torch.utils.data.DataLoader(
        CIFARTEST[str(params.num_classes)],
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=8,
    )

    # now = datetime.now()
    # dt_string = now.strftime("%H:%M:%S")
    # params.model_path = SAVED_MODELS_DIR / (params.run_name + dt_string + ".pth")
    params.model_path = SAVED_MODELS_DIR / (params.run_name + ".pth")

    # TRAINING
    wandb_init(params)
    train_model(params)


if __name__ == "__main__":
    main()
