"""
Replication of the case study "MNIST" training
from the master thesis Neural geodesic flows published at
https://doi.org/10.3929/ethz-b-000733724
"""

import wandb

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    NN_MNIST_encoder,
    NN_classification_decoder,
    NN_metric_regularized,
)

#get the relevant loss functions
from core.losses import (
    classification_loss,
    classification_error
)

#get the relevant utility methods
from applications.utils import (
    perform_training
)

#get the relevant methods to configure hyperparameters
from applications.configs import (
    get_wandb_config,
    PATH_LOGS
)


#variables that are identitical for each model
g_initializer = NN_metric_regularized

train_loss_function = classification_loss
test_loss_function = classification_error

################################ Training of the PyTorch inspired model with geodesic solve ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: MNIST",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "MNIST_train",
                          test_dataset_name = "MNIST_test",
                          model_name = "master_thesis/classification_model-geodesic-solve",
                          dim_dataspace = (1,28,28),
                          dim_M = 10,
                          psi_arguments = {},
                          phi_arguments = {'dim_M':10},
                          g_arguments = {'dim_M':10,
                                         'hidden_sizes':[256,256]},
                          train_dataset_size = 1000,
                          test_dataset_size = 100,
                          batch_size = 200,
                          loss_print_frequency = 1,
                          epochs = 10)

psi_initializer = NN_MNIST_encoder
phi_initializer = NN_classification_decoder

perform_training(config,
                 psi_initializer,
                 phi_initializer,
                 g_initializer,
                 train_loss_function,
                 test_loss_function)
