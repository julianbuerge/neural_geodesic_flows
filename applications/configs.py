"""
Flexibly store all parameters and hyperparameters of the model and training
in a wandb.config variable. Meant to be called from the srcipt that initializes
and executes a training run.

Define variables containing the paths to data
"""

import jax

import optax

import wandb

from pathlib import Path


#set all hyperparameters and other high level parameters and return them in a wandb.config variable
def get_wandb_config(train_dataset_name,
                     test_dataset_name,
                     model_name,
                     dim_dataspace,
                     dim_M,
                     phi_arguments,
                     psi_arguments,
                     g_arguments,
                     batch_size, #batching only happens on the train dataset
                     train_dataset_size = None, #if left as None it will load the whole dataset
                     test_dataset_size = None, #if left as None it will load the whole dataset
                     optimizer_name = "adam", learning_rate = 1e-3, epochs = 100, loss_print_frequency = 10,
                     random_seed = 0,
                     continue_training = False,
                     updated_model_name = None,
                     save = True,
                     **kwargs):

    #initalize the config variable
    config = wandb.config

    #mandatory arguments
    config.train_dataset_name = train_dataset_name
    config.test_dataset_name = test_dataset_name
    config.model_name = model_name
    config.dim_dataspace = dim_dataspace
    config.dim_M = dim_M

    config.phi_arguments = phi_arguments
    config.psi_arguments = psi_arguments
    config.g_arguments = g_arguments

    config.batch_size = batch_size

    #optional arguments
    config.train_dataset_size = train_dataset_size
    config.test_dataset_size = test_dataset_size
    config.optimizer_name = optimizer_name
    config.learning_rate = learning_rate
    config.epochs = epochs
    config.loss_print_frequency = loss_print_frequency

    config.random_seed = random_seed

    config.continue_training = continue_training

    if updated_model_name is None and continue_training:
        config.updated_model_name = f"{model_name}_continued"
    elif continue_training:
        config.updated_model_name = updated_model_name

    config.save = save

    #add any additional parameters passed in via kwargs
    for key, value in kwargs.items():
        setattr(config, key, value)

    return config

#meant to be used to initialize an optimizer before training using the name and learning_rate from the config variable
def get_optimizer(name, learning_rate):

    if name == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {name} is not supported.")

    return optimizer



################### variables storing data paths, meant to be imported by other modules ##################

def find_project_root(marker=".git"):
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

PATH_PROJECT_ROOT = find_project_root()
PATH_DATASETS = PATH_PROJECT_ROOT/'data'/'datasets'
PATH_MODELS = PATH_PROJECT_ROOT/'data'/'models'
PATH_LOGS = PATH_PROJECT_ROOT/'data'/'logs'
