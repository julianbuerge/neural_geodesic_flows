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
    input_target_loss,
    trajectory_reconstruction_loss,
    trajectory_prediction_loss,
    trajectory_loss
)

from core.models import (
    TangentBundle,
)

from core.template_psi_phi_g_functions_analytical import (
    psi_S2_normal,
    phi_S2_normal,
    g_S2_normal,
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

def unit_test_input_target_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda inputs, targets, times : input_target_loss(tangentbundle = tangentbundle,
                                                            inputs =inputs, targets = targets,
                                                                times = times)

    printheading(unit_name="input_target_loss")

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


############################### Testing #####################################

unit_test_recon_loss()
unit_test_input_target_loss()
unit_test_traj_recon_loss()
unit_test_traj_pred_loss()
unit_test_traj_loss()
