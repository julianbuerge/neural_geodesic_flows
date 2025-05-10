"""
Application of neural geodesic flows:
- initializes wandb session
- loads training data
- trains a model
- saves the model

modify to preference:
- arguments of get_wandb_config
- initializers for psi, phi and g
- loss functions

"""
import jax
import jax.numpy as jnp

import optax
import equinox as eqx

import numpy as np

import wandb

import json

#get the relevant models
from core.models import (
    TangentBundle,
    Classification
)

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    identity_diffeomorphism,
    NN_diffeomorphism,
    NN_diffeomorphism_for_chart,
    NN_split_diffeomorphism,
    NN_linear_split_diffeomorphism,
    NN_Jacobian_split_diffeomorphism,
    NN_Jacobian_split_diffeomorphism_for_chart,
    NN_conv_diffeomorphism_for_chart,
    NN_conv_diffeomorphism_for_parametrization,
    NN_pytorches_MNIST_encoder,
    identity_metric,
    NN_metric,
    NN_metric_regularized,
)

#get the relevant loss functions
from core.losses import (
    reconstruction_loss,
    prediction_reconstruction_loss,
    trajectory_reconstruction_loss,
    trajectory_prediction_loss,
    trajectory_loss,
    classification_loss,
    classification_error,
)

#get the relevant generic training methods
from core.training import (
    train,
)
#get the relevant methods for loading/saving data and models
from applications.utils import (
    create_dataloader,
    load_model,
    save_model,
)

#get the relevant methods to configure hyperparameters
from applications.configs import (
    get_wandb_config,
    get_optimizer,
    PATH_LOGS
)


################################ initialize a Weights and Biases project ################################
wandb.init(project="Neural geodesic flows",
           group = "Tests",
           dir=PATH_LOGS)

################################ choose all hyperparameters and neural networks used ################################

#get a wandb.config variable holding all hyper and high level parameters for the training run
#mandatory arguments: train/test_dataset_name, model_name, dim_dataspace, dim_M, phi/psi/g_arguments, batch_size
"""choose here!"""
config = get_wandb_config(train_dataset_name  = "half-sphere_trajectories_train",
                          test_dataset_name = "half-sphere_trajectories_test",
                          model_name = "half-sphere",
                          dim_dataspace = 6,
                          dim_M = 2,
                          psi_arguments = {"in_size": 6,
                                           "out_size": 4,
                                           "hidden_sizes": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 6,
                                           "hidden_sizes": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 25,
                          train_dataset_size = 100,
                          test_dataset_size = 30,
                          learning_rate = 1e-3,
                          epochs = 10, loss_print_frequency = 1,
                          continue_training = False,
                          updated_model_name = "",
                          save = True)

"""choose here!"""
psi_initializer = NN_diffeomorphism_for_chart
phi_initializer = NN_diffeomorphism
g_initializer = NN_metric_regularized

#above, choose the type of neural networks used for phi,psi, g. They have to have exactly two arguments which is a dictionary,
#which also has to be saved as a member variable, and a random key.
#they will get passed the dictionary specified above in the get config variable.
#if doing continued training of a saved model, assign the initializers of the networks that the model previously used.
#if you forgot, the network class names are written in the model_name_high_level_params.json file (for this exact purpose)

"""choose here!"""
train_loss_function = trajectory_loss
test_loss_function = trajectory_loss

#make sure that the chosen loss functions match the used datasets
#meaning they either take arguments & have keys
#inputs, targets, times
#or trajectories, times.

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

    key, key_phi, key_psi, key_g, key_model = jax.random.split(key, 5)

    phi_NN = phi_initializer(config.phi_arguments, key = key_phi)

    psi_NN = psi_initializer(config.psi_arguments, key = key_psi)

    g_NN = g_initializer(config.g_arguments, key = key_g)

    model = TangentBundle(dim_dataspace = config.dim_dataspace, dim_M = config.dim_M,
                            phi = phi_NN, psi = psi_NN, g = g_NN)


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
    else:
        save_model(model, config.model_name)

wandb.finish()

print("\nFinished training")
