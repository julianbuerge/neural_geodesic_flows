"""
Unit tests for losses.py
"""

import jax
import jax.numpy as jnp

from tests.utils import (
    printheading,
    print_function_evaluation,
    test_function_dimensionality,
    test_function_evaluation,
    test_metric_evaluation,
)


from core.losses import (
    reconstruction_loss,
    prediction_reconstruction_loss,
    trajectory_reconstruction_loss,
    trajectory_prediction_loss,
    trajectory_loss,
    classification_loss,
    classification_error,
)

from core.models import (
    TangentBundle,
)

from core.template_psi_phi_g_functions_analytical import (
    psi_S2_normal,
    phi_S2_normal,
    g_S2_normal,
)

from core.template_psi_phi_g_functions_neural_networks import (
    NN_diffeomorphism,
    NN_classification_decoder,
    NN_metric,
)


def unit_test_recon_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda inputs, targets, times : reconstruction_loss(tangentbundle = tangentbundle,
                                                            inputs =inputs, targets = targets,
                                                                times = times)

    printheading(unit_name="reconstruction_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,6),(100,6),(100,)])

def unit_test_pred_recon_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda inputs, targets, times : prediction_reconstruction_loss(tangentbundle = tangentbundle,
                                                            inputs =inputs, targets = targets,
                                                                times = times)

    printheading(unit_name="prediction_reconstruction_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,6),(100,6),(100,)])


def unit_test_traj_recon_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda trajectories, times : trajectory_reconstruction_loss(tangentbundle = tangentbundle,
                                                   trajectories = trajectories,
                                                   times = times)

    printheading(unit_name="trajectory_reconstruction_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,20,6),(100,20)])

def unit_test_traj_pred_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda trajectories, times : trajectory_prediction_loss(tangentbundle = tangentbundle,
                                                   trajectories = trajectories,
                                                   times = times)

    printheading(unit_name="trajectory_prediction_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,21,6),(100,21)])

def unit_test_traj_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda trajectories, times : trajectory_loss(tangentbundle = tangentbundle,
                                                   trajectories = trajectories,
                                                   times = times)

    printheading(unit_name="trajectory_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,21,6),(100,21)])


def unit_test_classif_loss(seed=0):

    psi_NN = NN_diffeomorphism({'in_size' : 14,
                                'out_size' : 6,
                                'hidden_sizes' : [16,16]})

    phi_NN = NN_classification_decoder({'dim_M':3})

    g_NN = NN_metric({'dim_M' : 3,
                      'hidden_sizes' : [16,16]})

    tangentbundle = TangentBundle(dim_dataspace = 14, dim_M = 3,
                                  psi = psi_NN, phi = phi_NN, g = g_NN)

    loss_dim = lambda inputs, targets, times : classification_loss(model = tangentbundle,
                                                            inputs = inputs,
                                                            targets = targets,
                                                            times = times)

    loss_eval = lambda inputs : classification_loss(model = tangentbundle,
                                        inputs = inputs,
                                        targets = jnp.ones((100,3)),
                                        times = jnp.ones((100,)))

    printheading(unit_name="classification_loss for randomdly initialized model Classification")

    test_function_dimensionality(func = loss_dim, in_shapes = [(100,14),(100,3),(100,)])

    print_function_evaluation(func = loss_eval, in_shapes = [(100,14)],seed=seed)

def unit_test_classif_error(seed=0):

    psi_NN = NN_diffeomorphism({'in_size' : 14,
                                'out_size' : 6,
                                'hidden_sizes' : [16,16]})

    phi_NN = NN_classification_decoder({'dim_M':3})

    g_NN = NN_metric({'dim_M' : 3,
                      'hidden_sizes' : [16,16]})

    tangentbundle = TangentBundle(dim_dataspace = 14, dim_M = 3,
                                  psi = psi_NN, phi = phi_NN, g = g_NN)

    loss_dim = lambda inputs, targets, times : classification_error(model = tangentbundle,
                                                            inputs = inputs,
                                                            targets = targets,
                                                            times = times)

    #the random inputs will have these fixed targets. So the resulting error is also random
    loss_eval = lambda inputs : classification_error(model = tangentbundle,
                                        inputs = inputs,
                                        targets = jnp.array([[0,1,0],
                                                             [0,0,1],
                                                             [1,0,0],
                                                             [1,0,0]]),
                                        times = jnp.ones((4,)))

    printheading(unit_name="classification_error for randomdly initialized model Classification")

    test_function_dimensionality(func = loss_dim, in_shapes = [(100,14),(100,3),(100,)])

    print_function_evaluation(func = loss_eval, in_shapes = [(4,14)],seed=seed)


############################### Testing #####################################

unit_test_recon_loss()
unit_test_pred_recon_loss()
unit_test_traj_recon_loss()
unit_test_traj_pred_loss()
unit_test_traj_loss()
unit_test_classif_loss(seed=0)
unit_test_classif_error(seed=0)
