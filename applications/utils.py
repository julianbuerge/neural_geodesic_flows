"""
Collection of auxiliarly methods:

- to load datasets
- to create dataloaders
- to save & load models
"""
import jax
import jax.numpy as jnp

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

        print(f"\nLoaded full dataset {name} of size {full_size} ({mode})\n")

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
