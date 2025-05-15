"""
Contains methods generic to training, namely:
- update method, updating a model based on its loss over a training batch
- train method, iterating through training epochs and calling the update method.

Specific setup on how data and models get loaded/saved as well as hyperparameter management is not
handled here. The idea is that one could do this their prefered way.
Therefore we don't do it here in core/ but instead in applications/.

the train method requires a wandb session to have been initialized.

the update method is the most highlevel method that get called repeatedly,
therefore we do jit (just in time compilation) on it.
"""

import jax
import jax.numpy as jnp

import equinox as eqx
import optax

import wandb

@eqx.filter_jit
def update(model, loss_function, optimizer, opt_state, batch):

    #compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(loss_function)(model, *batch)

    #apply optimizer update
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def train(model,
          train_loss_function,
          test_loss_function,
          train_dataloader,
          test_dataloader,
          optimizer,
          epochs,
          loss_print_frequency):

    print(f"Starting training of {epochs} epochs")

    #initalize optimizer state with current model parameters
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    #jit the test loss
    test_loss_function = eqx.filter_jit(test_loss_function)

    #training loop
    for epoch in range(epochs):

        epoch_loss = 0
        test_loss = 0

        #iterate over the batches in training DataLoader
        for batch in train_dataloader:

            #convert each element in the batch tuple to JAX arrays
            batch = tuple(jnp.array(tensor.numpy()) for tensor in batch)

            #update the model according to a training step
            model, opt_state, loss = update(model,
                                            train_loss_function,
                                            optimizer,
                                            opt_state,
                                            batch)

            #for logging we look at the loss in a single gradient descend step
            wandb.log({"gradient descend step loss": loss})

            #for printing we will look at the average loss in the epoch
            epoch_loss += loss


        #iterate over the batches in testing DataLoader
        for batch in test_dataloader:

            #convert each element in the batch tuple to JAX arrays
            batch = tuple(jnp.array(tensor.numpy()) for tensor in batch)

            #calculate the test loss on the batch
            loss = test_loss_function(model, *batch)

            #add batch loss to the total test loss
            test_loss += loss




        #average the losses over batches
        epoch_loss /= (len(train_dataloader))
        test_loss /= (len(test_dataloader))

        wandb.log({"test loss": test_loss})

        #print losses
        if epoch == 0 or (epoch + 1) % loss_print_frequency == 0:
            print(f"Epoch {epoch + 1}, Train loss: {epoch_loss}, Test loss: {test_loss}")

    return model
