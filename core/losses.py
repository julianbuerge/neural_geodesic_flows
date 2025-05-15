"""
Collection of loss functions
"""

import jax
import jax.numpy as jnp

#expect data of shape (batch_size, mathematical dimension), (batch_size,mathematical dimension), (batch_size)
def reconstruction_loss(tangentbundle, inputs, targets, times):

    #vectorize the functions from the tangentbundle
    encoder = jax.vmap(tangentbundle.psi, in_axes = 0)
    decoder = jax.vmap(tangentbundle.phi, in_axes = 0)

    #generate reconstructions
    reconstructions_inputs = decoder(encoder(inputs))
    reconstructions_targets = decoder(encoder(targets))

    #measure the quality of the reconstruction by MSE
    reconstructive_power = jnp.mean((reconstructions_inputs - inputs)**2) + jnp.mean((reconstructions_targets - targets)**2)

    #loss
    return reconstructive_power

#expect data of shape (batch_size, mathematical dimension), (batch_size,mathematical dimension), (batch_size)
def input_target_loss(tangentbundle, inputs, targets, times):

    #vectorize the functions from the tangentbundle
    exp = jax.vmap(tangentbundle.exp, in_axes = (0,0,None))
    encoder = jax.vmap(tangentbundle.psi, in_axes = 0)
    decoder = jax.vmap(tangentbundle.phi, in_axes = 0)

    #generate predictions
    num_steps = 49

    latent_inputs = encoder(inputs)
    latent_targets = encoder(targets)

    latent_predictions = exp(latent_inputs, times, num_steps)

    predictions = decoder(latent_predictions)

    #generate reconstructions
    reconstructions_inputs = decoder(latent_inputs)
    reconstructions_targets = decoder(latent_targets)

    #measure the quality of the predicition by MSE
    predictive_error = jnp.mean((predictions - targets)**2)
    latent_predictive_error = jnp.mean((latent_predictions - latent_targets)**2)

    #measure the quality of the reconstruction by MSE
    reconstructive_error = jnp.mean((reconstructions_inputs - inputs)**2 + (reconstructions_targets - targets)**2)

    #loss as a weighted combination (as they have the same units the weight should be in [0,1])
    return reconstructive_error + predictive_error + latent_predictive_error

#expect data of shape (batch_size, time steps, mathematical dimension), (batch_size,time steps)
def trajectory_reconstruction_loss(tangentbundle, trajectories, times):

    #vectorize the functions from the tangentbundle
    encode_trajectory = jax.vmap(tangentbundle.psi, in_axes = 0)
    decode_trajectory = jax.vmap(tangentbundle.phi, in_axes = 0)

    encode_many_trajectories = jax.vmap(encode_trajectory, in_axes = 0)
    decode_many_trajectories = jax.vmap(decode_trajectory, in_axes = 0)

    #generate reconstructions
    reconstructions = decode_many_trajectories(encode_many_trajectories(trajectories))

    #measure the quality of the reconstruction
    reconstructive_error = jnp.mean((reconstructions - trajectories)**2)

    #loss
    return reconstructive_error

#expect data of shape (batch_size, time steps, mathematical dimension), (batch_size,time steps)
def trajectory_prediction_loss(tangentbundle, trajectories, times):

    #find the final times and the number of steps to take (assume times are equidistant)
    final_times = times[:,-1]
    num_steps = times.shape[1] - 1 #time steps = num_steps + 1 (the first time step will be the initial and for each num step we go forward by one step)

    #vectorize the functions from the tangentbundle.

    #expect to be given a trajectory (num_steps + 1, math dim)
    encode_trajectory = jax.vmap(tangentbundle.psi, in_axes = 0)
    #expect to be given a batch of trajectories (many, num steps + 1, math dim)
    encode_many_trajectories = jax.vmap(encode_trajectory, in_axes = 0)

    #expect to be given a batch of initial points (many, math dim)
    encode_initial = jax.vmap(tangentbundle.psi, in_axes = 0)

    #expect to be given a batch of encoded initial points (many, math dim)
    find_geodesic = jax.vmap(tangentbundle.exp_return_trajectory, in_axes = (0,0,None))

    #expect to be given a geodesic (num steps + 1, math dim)
    decode_geodesic = jax.vmap(tangentbundle.phi, in_axes = 0)
    #expect to be given a batch of geodesics (many, num steps +1 , math dim)
    decode_many_geodesics = jax.vmap(decode_geodesic, in_axes = 0)


    #generate the predicted trajectories (which are geodesics) and match them
    #with the given trajectories in both data and latent space.

    #encode the given trajectories
    encoded_trajectories = encode_many_trajectories(trajectories)

    #encode all the initial points from the given trajectories, get shape (many, math dim)
    encoded_initial_points = encode_initial(trajectories[:,0,...])

    #find all the corresponding geodesics, get shape (many, num steps +1, math dim)
    geodesics = find_geodesic(encoded_initial_points, final_times, num_steps)

    #decode all the geodesics, get shape (many, num_steps + 1, math dim)
    decoded_geodesics = decode_many_geodesics(geodesics)

    #measure the deviation of the predicted versus the given trajectory in latent space
    predictive_error_latentspace = jnp.mean((encoded_trajectories - geodesics)**2)

    #measure the deviation of the predicted versus the given trajectory in dataspace
    predictive_error_dataspace = jnp.mean((trajectories - decoded_geodesics)**2)

    return predictive_error_dataspace + predictive_error_latentspace

def trajectory_loss(tangentbundle, trajectories, times):

        #find the predictive error (will be latent + dataspace)
        predictive_error = trajectory_prediction_loss(tangentbundle, trajectories, times)

        #find the reconstructive error
        reconstructive_error = trajectory_reconstruction_loss(tangentbundle, trajectories, times)

        #return the sum of prediction and reconstruction error
        return predictive_error + reconstructive_error
