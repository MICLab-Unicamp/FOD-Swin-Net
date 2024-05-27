import wandb


class ToolsWandb:

    @staticmethod
    def config_flatten(config, fconfig):
        for key in config:
            if isinstance(config[key], dict):
                fconfig = ToolsWandb.config_flatten(config[key], fconfig)
            else:
                fconfig[key] = config[key]
        return fconfig

    @staticmethod
    def init_wandb_run(f_configurations,
                       name_project="wileC_free_datasets",
                       reinit=True,
                       notes="Testing wandb implementation",
                       entity="oliveira_mats"):

        return wandb.init(project=name_project,
                          reinit=reinit,
                          config=f_configurations,
                          notes=notes,
                          entity=entity)

    # wandb.log({'epoch_train_acc': train_acc, 'epoch_train_loss': train_loss,
    #            'epoch_valid_acc': valid_acc, 'epoch_valid_loss': valid_loss})
