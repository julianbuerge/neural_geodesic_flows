"""
Collection of auxiliarly methods:

- to load datasets
- to create dataloaders
- to save & load models
- to do model training with these methods for data/model managment
- to do model inference with these methods for data/model managment
"""

import jax
import jax.numpy as jnp

import optax
import equinox as eqx

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import wandb

import json

from core.models import TangentBundle

#load variables with the relevant paths
from applications.configs import PATH_DATASETS
from applications.configs import PATH_MODELS

from applications.configs import (
    get_optimizer
)

from core.training import (
    train
)

#get the relevant inference methods
from core.inference import (
    input_target_model_analyis,
    trajectory_model_analyis
)


#load a dataset stored in PATH_DATASETS/name.npz and return it as three jax arrays.
#Optionally shrink the size by random selection or truncation
def load_dataset(name, size=None, random_selection=False, key=jax.random.PRNGKey(0)):

    path = PATH_DATASETS/f"{name}.npz"

    loaded_data = np.load(path)

    keys = set(loaded_data.files)

    if {"inputs", "targets", "times"} <= keys:
        mode = "input-target"

        #convert to jax arrays.
        arrays = tuple(jnp.array(loaded_data[k]) for k in ["inputs", "targets", "times"])

    elif {"trajectories", "times"} <= keys:
        mode = "trajectory"

        #convert to jax arrays.
        arrays = tuple(jnp.array(loaded_data[k]) for k in ["trajectories", "times"])

    else:
        raise ValueError(f"Dataset {name} has unrecognized keys: {loaded_data.files}")

    #perform the potential shrinking
    full_size = arrays[0].shape[0]

    #in case we don't want to shrink or specified as size bigger than the dataset, return the whole dataset (do nothing)
    if size is None or size >= full_size:

        print(f"\nLoaded full dataset {name} of type '{mode}' of size {full_size}\n")

    #else if we do want to shrink and passed a size smaller than the dataset, shrink it
    else:
        #shrink with random selection
        if random_selection:
            indices = jax.random.choice(key, full_size, (size,), replace=False)
        #or with truncation
        else:
            indices = jnp.arange(size)

        #actual shrinking
        arrays = tuple(arr[indices, ...] for arr in arrays)

        print(f"\nLoaded dataset {name} of type '{mode}' of shrunk size {size}.\n")

    return arrays, mode

#create a dataloader for a dataset stored in PATH_DATASETS/dataset_name.
def create_dataloader(dataset_name, batch_size, dataset_size=None, random_selection=False, key=jax.random.PRNGKey(0)):

    #convert jax arrays to pytorch tensors
    arrays, mode = load_dataset(name=dataset_name, size=dataset_size, random_selection=random_selection, key=key)
    tensors = tuple(torch.tensor(np.array(arr)) for arr in arrays)

    #account for potential batch case == None (test dataloaders might have this)
    if batch_size is None:
        batch_size = tensors[0].shape[0]

    #create a tensor dataset and a torch DataLoader
    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nCreated DataLoader for dataset {dataset_name} of type '{mode}' with batch size {batch_size}\n")

    return dataloader


#load a model of the type TangentBundle stored in PATH_MODELS/model_name.eqx and return it.
#Here we are using the models high level parameters stored
#in PATH_MODELS/model_name_high_level_params.json in order to load it correctly.
#You have to pass the initializers (just pass the classes) of psi, phi and g.
#that were used in training. If you forgot, their names are written in the json file (for this purpose).
def load_model(model_name, psi_initializer, phi_initializer, g_initializer):

    #paths of saved model and high level parameters
    model_path = PATH_MODELS/f"{model_name}.eqx"
    model_high_level_params_path = PATH_MODELS/f"{model_name}_high_level_params.json"

    #load the model high level parameters from the json file
    with open(model_high_level_params_path, 'r') as f:
        model_high_level_params = json.load(f)

    #initialize correct type neural networks (could also be hardcoded function, dependinging on the initializer)
    psi_NN = psi_initializer(model_high_level_params['psi_arguments'])

    phi_NN = phi_initializer(model_high_level_params['phi_arguments'])

    g_NN = g_initializer(model_high_level_params['g_arguments'])

    #using the models high level parameters create an instance of the exact same form
    model_prototype = TangentBundle(dim_dataspace = model_high_level_params['dim_dataspace'],
                                    dim_M = model_high_level_params['dim_M'],
                                    psi = psi_NN, phi = phi_NN, g = g_NN)

    #initialize the saved model
    model = eqx.tree_deserialise_leaves(model_path, like = model_prototype)

    print(f"\nLoaded model {model_name}\n")

    return model


#Takes a trained model of type TangentBundle and stores it and its high level parameters
#under PATH_MODELS/model_name.eqx and PATH_MODELS/model_name_high_level_params.json
def save_model(model, model_name):

    #obtain model high level parameters (such as dim_M ...)
    model_high_level_params = model.get_high_level_parameters()

    #store the trained model locally in Models
    eqx.tree_serialise_leaves(PATH_MODELS/f"{model_name}.eqx", model)

    #store the model parameters locally in Models
    with open(PATH_MODELS/f"{model_name}_high_level_params.json", 'w') as f:
        json.dump(model_high_level_params, f)

    print(f"\nSaved model under the name {model_name}\n")


#perform the training of a model with the loading and saving methods defined above
#and with weight and biases for hyperparameter managment.
#return the model.
def perform_training(config,
                    psi_initializer,
                    phi_initializer,
                    g_initializer,
                    train_loss_function,
                    test_loss_function):

    #update the run name
    if config.continue_training:
        wandb.run.name = config.updated_model_name
        print(f"\nContinuing run {config.model_name} as {config.updated_model_name}")
    else:
        wandb.run.name = config.model_name
        print(f"\nCommencing run {config.model_name}")

    #initialize top level random key, all others will be splits of this
    top_level_key = jax.random.PRNGKey(config.random_seed)

    ################################ load the problems dataset ################################
    key, key_train_loader, key_test_loader = jax.random.split(top_level_key, 3)

    train_dataloader = create_dataloader(dataset_name = config.train_dataset_name,
                                 batch_size = config.batch_size,
                                 dataset_size = config.train_dataset_size,
                                 random_selection = True,
                                 key = key_train_loader)

    test_dataloader = create_dataloader(dataset_name = config.test_dataset_name,
                                 batch_size = config.test_dataset_size,
                                 dataset_size = config.test_dataset_size,
                                 random_selection = True,
                                 key = key_test_loader)


    ################################ initialize a model ################################
    if config.continue_training:

        model = load_model(config.model_name,
                            psi_NN_initializer = psi_initializer,
                            phi_NN_initializer = phi_initializer,
                            g_NN_initializer = g_initializer)
    else:

        key, key_psi, key_phi, key_g = jax.random.split(key, 4)

        psi_NN = psi_initializer(config.psi_arguments, key = key_psi)

        phi_NN = phi_initializer(config.phi_arguments, key = key_phi)

        g_NN = g_initializer(config.g_arguments, key = key_g)

        model = TangentBundle(dim_dataspace = config.dim_dataspace, dim_M = config.dim_M,
                                psi = psi_NN, phi = phi_NN, g = g_NN)


    ################################ train the model ################################
    optimizer = get_optimizer(name = config.optimizer_name, learning_rate = config.learning_rate)

    #train the model
    model = train(model = model,
                  train_loss_function = train_loss_function,
                  test_loss_function = test_loss_function,
                  train_dataloader = train_dataloader,
                  test_dataloader = test_dataloader,
                  optimizer = optimizer,
                  epochs = config.epochs,
                  loss_print_frequency = config.loss_print_frequency)


    ################################ save the model ################################
    if config.save:
        if config.continue_training:

            save_model(model, config.updated_model_name)
            wandb.finish()
            print(f"\nFinished training model {config.updated_model_name}.")

        else:
            save_model(model, config.model_name)
            wandb.finish()
            print(f"\nFinished training model {config.model_name}.")

    return model


def perform_inference(model_name,
                      psi_initializer,
                      phi_initializer,
                      g_initializer,
                      dataset_name,
                      dataset_size,
                      seed = 0):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)

    #load the data, respecting the correct mode
    data, mode = load_dataset(name = dataset_name,
                 size=dataset_size,
                 random_selection=True,
                 key=jax.random.PRNGKey(seed))

    if mode == "input-target":
        inputs, targets, times = data

    elif mode =="trajectory":
        trajectories, times = data


    #perform the analysis respecting the correct mode
    if mode == "input-target":
        input_target_model_analyis(model, inputs, targets, times)

    elif mode =="trajectory":
        trajectory_model_analyis(model, trajectories, times)
