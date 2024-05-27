import argparse
import gc
import os
import shutil
import wandb
from factory_organize_classes import FactoryOrganize
from tqdm import trange
from constants import *
from utils.tools_wandb import ToolsWandb
from utils.util import *
from hydra import initialize, compose
from utils.read_dataset import ReadDataset
from change_pacient_on_the_fly import ChangePacientOnTheFly


def end_loss(dir_model, current_valid_loss,
             epoch, model, optimizer, criterion, name_model, run=None):
    best_valid_loss = current_valid_loss
    print(f"end_loss_: {best_valid_loss}")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, f'{dir_model}{name_model}')


def read_yaml_hydra(config_path=".", config_name="config"):
    with initialize(config_path=config_path, version_base="1.2"):
        cfg = compose(config_name=config_name)
        result = cfg.copy()
        return result


change_pacient_ont_the_fly = ChangePacientOnTheFly()

def run_train_epoch(model, optimizer, criterion, loader,
                    monitoring_metrics, epoch, scheduler, run, step_update=1000):
    model.to(DEVICE)
    model.train()
    torch.cuda.empty_cache()
    gc.collect()

    epoch_loss = 0

    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            inputs, labels = sample_batch["fodlr"], sample_batch["fodgt"]

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            pred_labels = model(inputs)

            loss = criterion(pred_labels, labels)

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item()}'
            )

            loss.backward()
            optimizer.step()

            if configs['wandb']:
                # wandb.log({'train_loss': loss})
                wandb.log({'train_loss': loss.item()})

            if configs["break_inside_train"]["type"] and configs["break_inside_train"]["iterator"] == batch_idx:
                break

        epoch_loss = (epoch_loss / len(loader))

    return epoch_loss


def run_validation(model, optimizer, criterion, loader,
                   epoch, configs, run, epsilon=1e-5):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(DEVICE)
        model.eval()
        running_loss = 0

        with trange(len(loader), desc='Validation Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, loader):
                optimizer.zero_grad()

                inputs, labels = sample_batch["fodlr"], sample_batch["fodgt"]

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                pred_labels = model(inputs)

                loss = criterion(pred_labels, labels)

                running_loss += loss.item()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch + 1}] Loss: {loss.item():.6f}"
                )

                if configs["break_inside_train"]["type"] and configs["break_inside_train"]["iterator"] == batch_idx:
                    break

        epoch_loss = (running_loss / len(loader))

    name_model = f"{configs['path_save_model']}{configs['network']}_{configs['reload_model']['data']}.pt"

    if configs['wandb']:
        wandb.log({'mean_valid_loss': epoch_loss})

    print(f"validation mean_mse: {epoch_loss}")

    end_loss(save_best_model.dir_model, epoch_loss, batch_idx,
             model, optimizer, criterion, name_model.replace("/", "/end_loss"), run)

    return epoch_loss


def get_params_lr_scheduler(configs):
    scheduler_kwargs = configs["lr_scheduler"]["info"]
    scheduler_type = configs["lr_scheduler"]["scheduler_type"]
    return scheduler_type, scheduler_kwargs


def calculate_parameters(model):
    qtd_model = sum(p.numel() for p in model.parameters())
    print(f"quantidade de parametros: {qtd_model}")
    return


def run_training_experiment(model, training_loader, valid_loader, optimizer, custom_lr_scheduler,
                            criterion, scheduler, configs, run
                            ):
    os.makedirs(configs["path_save_model"], exist_ok=True)

    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "accuracy": {"train": [], "validation": []}
    }

    calculate_parameters(model)

    size_3d_patch = training_loader.dataset.size_3d_patch

    read_dataset = ReadDataset(size_3d_patch)

    for epoch in range(0, configs["epochs"] + 1):
        if configs["training_per_patches"] is False:
            pacient = epoch

            train_loader = change_pacient_ont_the_fly.apply(training_loader,
                                                            read_dataset,
                                                            pacient)

            if train_loader is None:
                continue

        elif configs["training_per_patches"] == "miclab_GPU":
            pacient = epoch

            train_loader = change_pacient_ont_the_fly.apply_many_subjects(training_loader,
                                                                          read_dataset)

            if train_loader is None:
                continue

        elif configs["training_per_patches"] == "h5py":
            train_loader = training_loader

        else:
            train_loader = change_pacient_ont_the_fly.iterator_on_the_fly(training_loader,
                                                                          read_dataset)

        train_loss = run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics,
            epoch, scheduler, run
        )

        if configs["training_per_patches"] is False:
            validation_loader = change_pacient_ont_the_fly.apply(valid_loader,
                                                                 read_dataset,
                                                                 pacient)

        elif configs["training_per_patches"] == "miclab_GPU":
            pacient = epoch

            validation_loader = change_pacient_ont_the_fly.apply_many_subjects(valid_loader,
                                                                               read_dataset)

            if validation_loader is None:
                continue

        elif configs["training_per_patches"] == "h5py":
            validation_loader = valid_loader

        else:
            validation_loader = change_pacient_ont_the_fly.iterator_on_the_fly(valid_loader,
                                                                               read_dataset)

            if validation_loader is None:
                continue

        valid_loss = run_validation(
            model, optimizer, criterion, validation_loader,
            epoch, configs, run
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fod attention net train")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    path_dir, name_file = split_path_and_file(args.config_file)

    configs = read_yaml(args.config_file)

    abs_extra_train = "<your_path_in_here>"
    ids_extra_train = [f"{abs_extra_train}/{id}" for id in os.listdir(abs_extra_train)]

    print("============ Delete .wandb path ============")
    try:
        shutil.rmtree("wandb/")
    except:
        print("especific directory wandb/")

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs, f_configurations)

    model, train_loader, validation_loader, \
        optimizer, criterion, scheduler = FactoryOrganize.experiment_factory(configs)

    if configs['reload_model']['type']:
        name_model = f"{configs['path_to_save_model']}/{configs['network']}_{configs['reload_model']['data']}.pt"

        load_dict = torch.load(name_model)

        model.load_state_dict(load_dict['model_state_dict'])

    if "random_dataset_split" in configs:
        random_class = FactoryRand.call_rand_split(configs["random_dataset_split"]["name"])
        configs["random_dataset_split"]["parameters"][
            "path_name_create"] = f"{configs['network']}_{configs['reload_model']['data']}"
        samples_train, samples_valid = random_class.path_dataset(**configs["random_dataset_split"]["parameters"])

        train_loader.dataset.data_list_id_ = samples_train + ids_extra_train
        print(f"new length train: {len(train_loader.dataset.data_list_id_)}")

        validation_loader.dataset.data_list_id_ = samples_valid

        train_loader.dataset.count_subjects = len(samples_train)
        validation_loader.dataset.count_subjects = len(samples_valid)

    run = None

    if configs['wandb']:
        run = wandb.init(project="fodf_interpolation_models",
                         reinit=True,
                         config=f_configurations,
                         notes="Running Training Model",
                         entity="oliveira_mats")

    run_training_experiment(
        model, train_loader, validation_loader, optimizer, None,
        criterion, scheduler, configs, run
    )

    torch.cuda.empty_cache()
    wandb.finish()
