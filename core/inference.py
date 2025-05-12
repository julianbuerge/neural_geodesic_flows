"""
Definition of several model inference methods.

The essential methods are apply_model_function and find_indices.

All others call them, and use their return values to generate statistics or visualizations.

We use three different formats of the data:
- inputs, targets & times,    expected to be of shapes (many, mathdim) & (many,)
- trajectories & times,       expected to be of shape (many, trajectory points, mathdim) & (many, trajectory points)
- points,                     expected to be of shape (many, mathdim)
"""

import jax
import jax.numpy as jnp

import numpy as np

from core.models import TangentBundle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull

#apply a method of the model to given input data, return the models outputs
#data is supposed to be a tuple of jax arrays, exactly as many as the arguments of model_function
def apply_model_function(model_function, data, vmap_axes):

    #vmap the function along the axis 0 for each input array in the data tuple
    function = jax.vmap(model_function, in_axes = vmap_axes)

    #apply the function with each array in the tuple data as an input
    outputs = function(*data)

    return outputs

#find the indices of the 10 (or size many) worst, average and best performing cases
def find_indices(correct_outputs, model_outputs, size = 10):

    #compute squared error across all but the "many" axis (axis 0)
    error_axes = tuple(range(1, correct_outputs.ndim))
    errors = jnp.mean((correct_outputs - model_outputs) ** 2, axis=error_axes)

    #find the size many indices with the smallest, average, and largest predictive error.
    sorted_indices = jnp.argsort(predictive_errors)

    best_indices = sorted_indices[:size]
    worst_indices = sorted_indices[-size:]
    avg_start = len(sorted_indices) // 2 - size // 2
    average_indices = sorted_indices[avg_start : avg_start + size]

    return worst_indices, average_indices, best_indices


#perform reconstruction and prediction analysis on input to target style data
def input_target_model_analyis(model, inputs, targets, times):

    #perform reconstruction on the whole trajectories
    autoencode = lambda y : model.phi(model.psi(y))

    recon_inputs = apply_model_function(autoencode, tuple((inputs,)), vmap_axes = (0))
    recon_targets = apply_model_function(autoencode, tuple((targets,)), vmap_axes = (0))

    #perform prediction starting from the inputs until the final times with 49 steps.
    pred = apply_model_function(model,
                                tuple((inputs, times, 49)),
                                vmap_axes = (0,0,None))

    #find the reconstruction and prediction errors
    recon_error = jnp.mean((inputs - recon_inputs)**2 + (targets - recon_targets)**2)
    pred_error = jnp.mean((targets - pred)**2)

    print(f"Reconstruction mean square error {recon_error}\n")
    print(f"Prediction mean square error {pred_error}\n")

#perform reconstruction and prediction analysis on trajectory style data
def trajectory_model_analyis(model, trajectories, times):

    #perform reconstruction on the whole trajectories
    autoencode = lambda y : model.phi(model.psi(y))
    autoencode_traj = jax.vmap(autoencode, in_axes = 0) #this will vmap along one trajectory

    recon = apply_model_function(autoencode_traj,
                                 tuple((trajectories,)),
                                 vmap_axes = (0)) #this will vmap along all trajectories

    #perform prediction starting from the initial points on the trajectories until the final times with time-points - 1 steps.
    pred = apply_model_function(model.get_geodesic,
                                tuple((trajectories[:,0,:], times[:,-1], times.shape[1] - 1)),
                                vmap_axes = (0,0,None))

    #find the reconstruction and prediction errors
    recon_error = jnp.mean((trajectories - recon)**2)
    pred_error = jnp.mean((trajectories - pred)**2)

    print(f"Reconstruction mean square error {recon_error}\n")
    print(f"Prediction mean square error {pred_error}\n")


#TO DO: Add general visual inference methods
def input_target_model_visualization(model, inputs, targets, times):
    pass

def trajectory_model_visualization(model, trajectories, times):
    pass

def curvature_visualization(model, points):
    pass
