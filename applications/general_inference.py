"""
Run model analysis (inference) methods.

Load a trained model and then run analysis functions,
passing a testdataset.

You only need to modify
"""
import os

import jax
import jax.numpy as jnp

import equinox as eqx

import numpy as np

import json

#customize the figure default style and format
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
})

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

#get some loading methods to load models and test datasets
from applications.utils import (
    load_dataset,
    load_model
)

#get the relevant inference methods
from core.inference import (
    input_target_model_analyis,
    trajectory_model_analyis
)

########################## initialize problem ##########################

dataset_name = "half-sphere_inputs-targets_test"
dataset_size = 128

#automatically load the data, respecting the correct mode
data, mode = load_dataset(name = dataset_name,
             size=dataset_size,
             random_selection=True,
             key=jax.random.PRNGKey(0))

if mode == "input-target":
    inputs, targets, times = data

elif mode =="trajectory":
    trajectories, times = data


########################## load a model ##########################

model_name = "half-sphere"

psi_initializer = NN_diffeomorphism_for_chart
phi_initializer = NN_diffeomorphism
g_initializer = NN_metric_regularized

#above assign the initializers of psi, phi and g that the model used,
#their names are written in the model_name_high_level_params.json file

model = load_model(model_name,
                   psi_initializer = psi_initializer,
                   phi_initializer = phi_initializer,
                   g_initializer = g_initializer)



########################## perform inference ##########################

#automatically perform the analysis respecting the correct mode
if mode == "input-target":
    input_target_model_analyis(model, inputs, targets, times)

elif mode =="trajectory":
    trajectory_model_analyis(model, trajectories, times)
