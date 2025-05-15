"""
Replication of the case study "MNIST" training
from the master thesis Neural geodesic flows published at
https://doi.org/10.3929/ethz-b-000733724
"""

import jax

import wandb


from core.models import TangentBundle

from applications.configs import (
    get_optimizer
)

from core.training import (
    train
)

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    NN_metric_regularized,
)

#get MNIST-classification-specialized templates and loss functions
from applications.master_thesis.classification_ngf_modification import (
    NN_MNIST_encoder,
    NN_MNIST_decoder,
    classification_loss,
    classification_error
)

#get the relevant utility methods
from applications.utils import (
    create_dataloader,
    save_model,
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
                          psi_arguments = {'out_size':20},
                          phi_arguments = {'dim_M':10},
                          g_arguments = {'dim_M':10,
                                         'hidden_sizes':[256,256]},
                          train_dataset_size = 60000,
                          test_dataset_size = 100,
                          batch_size = 1000,
                          loss_print_frequency = 1,
                          epochs = 10)

psi_initializer = NN_MNIST_encoder
phi_initializer = NN_MNIST_decoder

perform_training(config,
                 psi_initializer,
                 phi_initializer,
                 g_initializer,
                 train_loss_function,
                 test_loss_function)

################################ Training of the PyTorch inspired model without geodesic solve ################################
wandb.init(project="Neural geodesic flows",
           group = "Master thesis: MNIST",
           dir=PATH_LOGS)

config = get_wandb_config(train_dataset_name  = "MNIST_train",
                          test_dataset_name = "MNIST_test",
                          model_name = "master_thesis/classification_model-no-geodesic-solve",
                          dim_dataspace = (1,28,28),
                          dim_M = 10,
                          psi_arguments = {'out_size':10},
                          phi_arguments = {},
                          g_arguments = {},
                          train_dataset_size = 60000,
                          test_dataset_size = 100,
                          batch_size = 1000,
                          loss_print_frequency = 1,
                          epochs = 10)

psi_initializer = NN_MNIST_encoder

#now we customly call train, using psi as the model
print(f"\nCommencing run {config.model_name}")

top_level_key = jax.random.PRNGKey(config.random_seed)

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

key, key_psi = jax.random.split(key, 2)

model = psi_initializer(config.psi_arguments, key = key_psi)

optimizer = get_optimizer(name = config.optimizer_name, learning_rate = config.learning_rate)

model = train(model = model,
              train_loss_function = train_loss_function,
              test_loss_function = test_loss_function,
              train_dataloader = train_dataloader,
              test_dataloader = test_dataloader,
              optimizer = optimizer,
              epochs = config.epochs,
              loss_print_frequency = config.loss_print_frequency)

save_model(model, config.model_name)
wandb.finish()
print(f"\nFinished training model {config.model_name}.")
