"""
Replication of the case study "Toy problem on S^2_+" training
from the master thesis Neural geodesic flows published at
https://doi.org/10.3929/ethz-b-000733724
"""

import wandb

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    NN_diffeomorphism,
    NN_Jacobian_split_diffeomorphism,
    NN_metric,
)

#get the relevant loss functions
from core.losses import (
    input_target_loss,
    trajectory_loss,
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


################################ Training of model A ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: toy problem on S^2_+",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "half-sphere_inputs-targets_train",
                          test_dataset_name = "half-sphere_inputs-targets_test",
                          model_name = "master_thesis/toy-problem_model-A",
                          dim_dataspace = 6,
                          dim_M = 2,
                          psi_arguments = {"in_size": 6,
                                           "out_size": 4,
                                           "hidden_sizes_x": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 6,
                                           "hidden_sizes_x": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 512,
                          test_dataset_size = 1024,
                          epochs = 50)

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric

train_loss_function = input_target_loss
test_loss_function = input_target_loss

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)

################################ Training of model B ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: toy problem on S^2_+",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "half-sphere_inputs-targets_train",
                          test_dataset_name = "half-sphere_inputs-targets_test",
                          model_name = "master_thesis/toy-problem_model-B",
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
                          batch_size = 512,
                          test_dataset_size = 1024,
                          epochs = 100)

psi_initializer = NN_diffeomorphism
phi_initializer = NN_diffeomorphism
g_initializer = NN_metric

train_loss_function = input_target_loss
test_loss_function = input_target_loss

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)

################################ Training of model C ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: toy problem on S^2_+",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "half-sphere_trajectories_train",
                          test_dataset_name = "half-sphere_trajectories_test",
                          model_name = "master_thesis/toy-problem_model-C",
                          dim_dataspace = 6,
                          dim_M = 2,
                          psi_arguments = {"in_size": 6,
                                           "out_size": 4,
                                           "hidden_sizes_x": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 6,
                                           "hidden_sizes_x": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 512,
                          test_dataset_size = 1024,
                          epochs = 50)

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric

train_loss_function = trajectory_loss
test_loss_function = trajectory_loss

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)

################################ Training of model D ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: toy problem on S^2_+",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "half-sphere_trajectories_train",
                          test_dataset_name = "half-sphere_trajectories_test",
                          model_name = "master_thesis/toy-problem_model-D",
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
                          batch_size = 512,
                          test_dataset_size = 1024,
                          epochs = 100)

psi_initializer = NN_diffeomorphism
phi_initializer = NN_diffeomorphism
g_initializer = NN_metric

train_loss_function = trajectory_loss
test_loss_function = trajectory_loss

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)
