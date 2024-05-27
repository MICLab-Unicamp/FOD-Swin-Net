from constants import *


class FactoryOrganize:
    @staticmethod
    def get_dataset(dataset_configs, configs, train=True):
        try:
            transform_type = list(dataset_configs.values())[0]["transform"]
            if transform_type:
                transformations_configs = configs["transformation"]
                transfomations = FACTORY_DICT["transformation"][transform_type](
                    **transformations_configs[transform_type]
                )
                list(dataset_configs.values())[0]["transform"] = transfomations.get_transformations(train)
        except:
            pass

        dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
            **dataset_configs[list(dataset_configs.keys())[0]]
        )

        return dataset

    @staticmethod
    def set_samples_dataset(configs, samples, type_dataset='train_dataset', key_data="path_data"):
        configs[type_dataset][list(configs[type_dataset].keys())[0]][key_data] = samples
        return configs

    @staticmethod
    def set_length_dataset(configs, len_, type_dataset='train_dataset', key_data="length_dataset"):
        configs[type_dataset][list(configs[type_dataset].keys())[0]][key_data] = len_
        return configs

    @staticmethod
    def get_workers(dataset_configs):
        configs = dataset_configs[list(dataset_configs.keys())[0]]
        return configs["num_workers"]

    @staticmethod
    def experiment_factory(configs):
        train_dataset_configs = configs["train_dataset"]
        validation_dataset_configs = configs["valid_dataset"]
        model_configs = configs["model"]
        optimizer_configs = configs["optimizer"]
        criterion_configs = configs["loss"]

        # Construct the dataloaders with any given transformations (if any)
        train_dataset = FactoryOrganize.get_dataset(train_dataset_configs, configs, True)
        validation_dataset = FactoryOrganize.get_dataset(validation_dataset_configs, configs, False)

        train_workers = FactoryOrganize.get_workers(train_dataset_configs)
        valid_workers = FactoryOrganize.get_workers(validation_dataset_configs)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=configs["train"]["batch_size"], shuffle=True,
            num_workers=train_workers
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=configs["valid"]["batch_size"], shuffle=False,
            num_workers=valid_workers
        )

        # Build model
        if type(dict(model_configs)) == dict: # incoerencia
            model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
                **model_configs[list(model_configs.keys())[0]]
            )
        else:
            model = FACTORY_DICT["model"][model_configs]()

        optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
            model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
        )
        criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]]

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min'
        )

        return model, train_loader, validation_loader, optimizer, \
            criterion, scheduler
