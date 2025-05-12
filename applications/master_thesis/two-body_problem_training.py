"""
Replication of the case study "Two body problem" training
from the master thesis Neural geodesic flows published at
https://doi.org/10.3929/ethz-b-000733724
"""

import wandb

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    NN_diffeomorphism_for_chart,
    NN_Jacobian_split_diffeomorphism_for_chart,
    NN_Jacobian_split_diffeomorphism,
    NN_metric_regularized,
)

#get the relevant loss functions
from core.losses import (
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


#variables that are identitical for each model
g_initializer = NN_metric_regularized

train_loss_function = trajectory_loss
test_loss_function = trajectory_loss

################################ Training of model A ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: toy problem on S^2_+",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "two-body-problem_fixed-energy-trajectories_train",
                          test_dataset_name = "two-body-problem_fixed-energy-trajectories_test",
                          model_name = "master_thesis/two-body-problem_model-A",
                          dim_dataspace = 8,
                          dim_M = 2,
                          psi_arguments = {"in_size": 8,
                                           "out_size": 4,
                                           "hidden_sizes_x": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 8,
                                           "hidden_sizes_x": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 15,
                          epochs = 500)

psi_initializer = NN_Jacobian_split_diffeomorphism_for_chart
phi_initializer = NN_Jacobian_split_diffeomorphism

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)

################################ Training of model A ################################
config = get_wandb_config(train_dataset_name  = "two-body-problem_fixed-energy-trajectories_train",
                          test_dataset_name = "two-body-problem_fixed-energy-trajectories_test",
                          model_name = "master_thesis/two-body-problem_model-A",
                          dim_dataspace = 8,
                          dim_M = 2,
                          psi_arguments = {"in_size": 8,
                                           "out_size": 4,
                                           "hidden_sizes_x": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 8,
                                           "hidden_sizes_x": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 15,
                          loss_print_frequency = 50,
                          epochs = 500)

psi_initializer = NN_Jacobian_split_diffeomorphism_for_chart
phi_initializer = NN_Jacobian_split_diffeomorphism

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

config = get_wandb_config(train_dataset_name  = "two-body-problem_fixed-energy-trajectories_train",
                          test_dataset_name = "two-body-problem_fixed-energy-trajectories_test",
                          model_name = "master_thesis/two-body-problem_model-B",
                          dim_dataspace = 8,
                          dim_M = 2,
                          psi_arguments = {"in_size": 8,
                                           "out_size": 4,
                                           "hidden_sizes": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 8,
                                           "hidden_sizes": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 15,
                          loss_print_frequency = 50,
                          epochs = 500)

psi_initializer = NN_diffeomorphism_for_chart
phi_initializer = NN_diffeomorphism

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

config = get_wandb_config(train_dataset_name  = "two-body-problem_trajectories_train",
                          test_dataset_name = "two-body-problem_trajectories_test",
                          model_name = "master_thesis/two-body-problem_model-C",
                          dim_dataspace = 8,
                          dim_M = 2,
                          psi_arguments = {"in_size": 8,
                                           "out_size": 4,
                                           "hidden_sizes_x": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 8,
                                           "hidden_sizes_x": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 400,
                          test_dataset_size = 900,
                          loss_print_frequency = 50,
                          epochs = 500)

psi_initializer = NN_Jacobian_split_diffeomorphism_for_chart
phi_initializer = NN_Jacobian_split_diffeomorphism

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

config = get_wandb_config(train_dataset_name  = "two-body-problem_trajectories_train",
                          test_dataset_name = "two-body-problem_trajectories_test",
                          model_name = "master_thesis/two-body-problem_model-D",
                          dim_dataspace = 8,
                          dim_M = 2,
                          psi_arguments = {"in_size": 8,
                                           "out_size": 4,
                                           "hidden_sizes": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 8,
                                           "hidden_sizes": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 400,
                          test_dataset_size = 900,
                          loss_print_frequency = 50,
                          epochs = 500)

psi_initializer = NN_diffeomorphism_for_chart
phi_initializer = NN_diffeomorphism

perform_training(config,
                psi_initializer,
                phi_initializer,
                g_initializer,
                train_loss_function,
                test_loss_function)
