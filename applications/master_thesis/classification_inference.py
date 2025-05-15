"""
Replication of the case study "MNIST" inference
from the master thesis Neural geodesic flows published at
https://doi.org/10.3929/ethz-b-000733724
"""

import jax
import jax.numpy as jnp

import equinox as eqx

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    NN_metric_regularized,
)

#get MNIST-classification-specialized templates and loss functions
from applications.master_thesis.classification_ngf_modification import (
    NN_MNIST_encoder,
    NN_MNIST_decoder,
    classification_error
)

#get some loading methods to load models and test datasets
from applications.utils import (
    load_dataset,
    load_model,
)

from applications.configs import PATH_MODELS

def MNIST_accuracy_analysis(model):

    #load the data, respecting the correct mode
    data, _ = load_dataset(name = "MNIST_test", size = 10000)

    inputs, targets, times = data

    accuracy = 100*(1-classification_error(model, inputs, targets, times))

    print(f"Test accuracy of model {model_name} is {accuracy:.2f}%.")



################### model with geodesic solve ###################

model_name = "master_thesis/classification_model-geodesic-solve"

psi_initializer = NN_MNIST_encoder
phi_initializer = NN_MNIST_decoder
g_initializer = NN_metric_regularized

#load the model
model = load_model(model_name,
                   psi_initializer = psi_initializer,
                   phi_initializer = phi_initializer,
                   g_initializer = g_initializer)

MNIST_accuracy_analysis(model)

################### model without geodesic solve ###################

model_name = "master_thesis/classification_model-no-geodesic-solve"

psi_initializer = NN_MNIST_encoder

#we need to manually load the model, as it is not an instance of TangentBundle but instead of NN_MNIST_encoder
model_path = PATH_MODELS/f"{model_name}.eqx"

model_prototype = psi_initializer({'out_size':10})

model = eqx.tree_deserialise_leaves(model_path, like = model_prototype)

print(f"\nLoaded model {model_name}\n")


MNIST_accuracy_analysis(model)
