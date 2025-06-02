"""
Application of neural geodesic flows:
- initializes wandb session
- setup of a training run
- performs the training run (which saves the model if so specified)
"""

import wandb

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
    identity_metric,
    NN_metric,
    NN_metric_regularized,
)

#get the relevant loss functions
from core.losses import (
    reconstruction_loss,
    input_target_loss,
    trajectory_reconstruction_loss,
    trajectory_prediction_loss,
    trajectory_loss
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


################################ initialize a Weights and Biases project ################################
wandb.init(project="Neural geodesic flows",
           group = "Tests",
           dir=PATH_LOGS)

################################ choose all hyperparameters and neural networks used ################################

#get a wandb.config variable holding all hyper and high level parameters for the training run
#mandatory arguments: train/test_dataset_name, model_name, dim_dataspace, dim_M, psi/phi/g_arguments, batch_size

config = get_wandb_config(train_dataset_name  = "half-sphere_trajectories_train",
                          test_dataset_name = "half-sphere_trajectories_test",
                          model_name = "half-sphere-quick",
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
                          batch_size = 64,
                          train_dataset_size = 512,
                          test_dataset_size = 256,
                          learning_rate = 1e-3,
                          epochs = 10, loss_print_frequency = 1,
                          continue_training = False,
                          updated_model_name = "",
                          save = True)

psi_initializer = NN_diffeomorphism
phi_initializer = NN_diffeomorphism
g_initializer = NN_metric_regularized

#above, choose the type of neural networks used for psi,phi, g. They have to have two arguments which is a dictionary,
#which also has to be saved as a member variable, and a random key.
#they will get passed the dictionary specified above in the get config variable.
#if doing continued training of a saved model, assign the initializers of the networks that the model previously used.
#if you forgot, the network class names are written in the model_name_high_level_params.json file (for this exact purpose)

train_loss_function = trajectory_loss
test_loss_function = trajectory_loss

#make sure that the chosen loss functions match the used datasets
#meaning they either take arguments & have keys
#inputs, targets, times
#or trajectories, times.

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)
