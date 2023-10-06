import torch
import wandb
from src.models.utils import SaveBestModel, get_avg_loss, TensorCat
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler
from src.models.utils import get_stats, get_dict_stats
from datetime import datetime
from config.constants import SAVED_MODELS_DIR


def train(params):
    use_amp = True if params.device == "cuda:0" else False
    scaler = GradScaler(enabled=use_amp)
    params.model.train()
    tensor_cat = TensorCat()
    loss_sum = 0

    for idx, (X, y) in enumerate(params.training_set):
        X, y = X.to(params.device), y.to(params.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = params.model(X)
            loss = params.loss_fn(pred, y)
            loss = loss / params.accumulation

        # accumulate
        scaler.scale(loss).backward()

        if ((idx + 1) % params.accumulation == 0) or (
            idx + 1 == len(params.training_set)
        ):
            scaler.step(params.optimizer)
            scaler.update()
            params.optimizer.zero_grad()

        if params.toy_flag and idx == 1:
            print("TOYFLAG")
            break

        loss_sum += loss.item()
        tensor_cat(pred, y)

    return get_stats(
        tensor_cat.total_preds,
        tensor_cat.total_targets,
        get_avg_loss(loss_sum, params.training_set, params),
        params,
    )


def test(params, dataset):
    params.model.eval()
    loss_sum, correct = 0, 0
    tensor_cat = TensorCat()
    use_amp = params.device == "cuda:0"

    with torch.no_grad():
        for idx, (X, y) in enumerate(dataset):
            X, y = X.to(params.device), y.to(params.device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                pred = params.model(X)
                tensor_cat(pred, y)

            loss_sum += params.loss_fn(pred, y).item()

            if params.toy_flag and idx == 1:
                print("TOYFLAG")
                break

    return get_stats(
        tensor_cat.total_preds,
        tensor_cat.total_targets,
        loss_sum / len(dataset),
        params,
    )


def train_model(params):
    model_saver = SaveBestModel()
    scheduler = None
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    params.model_path = SAVED_MODELS_DIR / (params.run_name + dt_string + ".pt")

    if params.lr_search:
        scheduler = StepLR(params.optimizer, step_size=30, gamma=0.1)

    for epoch in range(params.epochs):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        params.lr = params.optimizer.param_groups[-1]["lr"]

        training_stats = train(params)
        print(f"Training| balanced accuracy: {training_stats['Balanced accuracy']}")
        # validation_stats = test(params, params.validation_set)

        test_stats = test(params, params.test_set)
        print(f"Test| balanced accuracy: {test_stats['Balanced accuracy']}")

        model_saver(test_stats["Balanced accuracy"], params, epoch)

        if params.lr_search:
            scheduler.step()

        if params.wandb_flag:
            wandb_data = [
                ("training", training_stats),
                # ("validation", validation_stats),
                ("test", test_stats),
            ]
            if epoch == 0:
                d = get_dict_stats(wandb_data)
                d.update(params.distributions)
                wandb.log(d)
            else:
                wandb.log(get_dict_stats(wandb_data))

        if params.toy_flag:
            break

    print("Done!")
