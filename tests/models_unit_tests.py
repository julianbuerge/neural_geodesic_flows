"""
Unit tests for models.py
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

from core.models import (
    TangentBundle,
    Classification,
)

from core.template_psi_phi_g_functions_analytical import (
    phi_S2_normal,
    psi_S2_normal,
    g_S2_normal,
    scalarproduct_S2_normal,
    connection_coeffs_S2_normal,
    geodesic_ODE_function_S2_normal,
    exp_S2_normal,
    exp_return_trajectory_S2_normal,
    get_geodesic_S2_normal,
    Riemann_curvature_S2_normal,
    Ricci_curvature_S2_normal,
    scalar_curvature_S2_normal,
    sectional_curvature_S2_normal
)


def unit_test_scalarproduct(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.scalarproduct")

    test_function_dimensionality(func = tangentbundle.scalarproduct, in_shapes = [(2,),(2,),(2,)])

    test_function_evaluation(func = tangentbundle.scalarproduct,
                                correct_func = scalarproduct_S2_normal, in_shapes = [(2,),(2,),(2,)],
                                    seed = seed)

def unit_test_connection_coeffs(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.connection_coeffs")

    test_function_dimensionality(func = tangentbundle.connection_coeffs, in_shapes = [(2,)])

    test_function_evaluation(func = tangentbundle.connection_coeffs,
                                correct_func = connection_coeffs_S2_normal, in_shapes = [(2,)],
                                    seed = seed)

def unit_test_geodesic_ODE_function(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.geodesic_ODE_function")

    test_function_dimensionality(func = tangentbundle.geodesic_ODE_function, in_shapes = [(4,)])

    test_function_evaluation(func = tangentbundle.geodesic_ODE_function,
                                correct_func = geodesic_ODE_function_S2_normal, in_shapes = [(4,)],
                                    seed = seed)

def unit_test_exp(seed=0):

    exp_correct = lambda z : exp_S2_normal(z,t=1,num_steps=10)

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    exp_bundle = lambda z : tangentbundle.exp(z,t=1,num_steps=10)

    printheading(unit_name="TangentBundle.exp")

    test_function_dimensionality(func = exp_bundle, in_shapes = [(4,)])

    test_function_evaluation(func = exp_bundle,
                                correct_func = exp_correct, in_shapes = [(4,)],seed=seed)

def unit_test_exp_return_trajectory(seed=0):

    exp_return_traj_correct = lambda z : exp_return_trajectory_S2_normal(z,t=1,num_steps=10)

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    exp_return_traj_bundle = lambda z : tangentbundle.exp_return_trajectory(z,t=1,num_steps=10)

    printheading(unit_name="TangentBundle.exp_return_trajectory")

    test_function_dimensionality(func = exp_return_traj_bundle, in_shapes = [(4,)])

    test_function_evaluation(func = exp_return_traj_bundle,
                                correct_func = exp_return_traj_correct, in_shapes = [(4,)],seed=seed)

def unit_test_get_geodesic(seed=0):

    get_geodesic_correct = lambda y : get_geodesic_S2_normal(y,t=1,num_steps=10)

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    get_geodesic_bundle = lambda y : tangentbundle.get_geodesic(y,t=1,num_steps=10)

    printheading(unit_name="TangentBundle.get_geodesic")

    test_function_dimensionality(func = get_geodesic_bundle, in_shapes = [(6,)])

    test_function_evaluation(func = get_geodesic_bundle,
                                correct_func = get_geodesic_correct, in_shapes = [(6,)],seed=seed)

def unit_test_TangentBundle(seed=0):

    num_steps = 10

    forward = lambda r,t : phi_S2_normal(exp_S2_normal(psi_S2_normal(r), t, num_steps))

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    bundle_call = lambda r,t : tangentbundle(r,t,num_steps)

    printheading(unit_name="TangentBundle.__call__")

    test_function_dimensionality(func = bundle_call, in_shapes = [(6,),()])

    test_function_evaluation(func = bundle_call,
                                correct_func = forward, in_shapes = [(6,),()],seed=seed)

def unit_test_Riemann_curvature(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.Riemann_curvature")

    test_function_dimensionality(func = tangentbundle.Riemann_curvature, in_shapes = [(2,)])

    test_function_evaluation(func = tangentbundle.Riemann_curvature,
                                correct_func = Riemann_curvature_S2_normal, in_shapes = [(2,)],
                                    seed = seed)

def unit_test_Ricci_curvature(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.Ricci_curvature")

    test_function_dimensionality(func = tangentbundle.Ricci_curvature, in_shapes = [(2,)])

    test_function_evaluation(func = tangentbundle.Ricci_curvature,
                                correct_func = Ricci_curvature_S2_normal, in_shapes = [(2,)],
                                    seed = seed)

def unit_test_scalar_curvature(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.scalar_curvature")

    test_function_dimensionality(func = tangentbundle.scalar_curvature, in_shapes = [(2,)])

    test_function_evaluation(func = tangentbundle.scalar_curvature,
                                correct_func = scalar_curvature_S2_normal, in_shapes = [(2,)],
                                    seed = seed)

def unit_test_sectional_curvature(seed=0):

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    printheading(unit_name="TangentBundle.sectional_curvature")

    test_function_dimensionality(func = tangentbundle.sectional_curvature, in_shapes = [(2,)])

    test_function_evaluation(func = tangentbundle.sectional_curvature,
                                correct_func = sectional_curvature_S2_normal, in_shapes = [(2,)],
                                    seed = seed)


def unit_test_Classification():

    t = 1
    num_steps = 10

    classes = 5

    nn_arguments = {'amount_classes' : classes,
                    'hidden_sizes' : [16,8]}

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    phi = phi_S2_normal, psi = psi_S2_normal,
                                        g = g_S2_normal)

    classification = Classification(tangentbundle, nn_arguments)

    classificaiton_call = lambda y : classification(y,t,num_steps)

    printheading(unit_name=f"Classification.__call__ for {classes} classes")

    test_function_dimensionality(func = classificaiton_call, in_shapes = [(6,)])

############################### Testing #####################################
unit_test_scalarproduct(seed=0)
unit_test_connection_coeffs(seed=0)
unit_test_geodesic_ODE_function(seed=0)
unit_test_exp(seed=0)
unit_test_exp_return_trajectory(seed=0)
unit_test_get_geodesic(seed=0)
unit_test_TangentBundle(seed=0) #WHY IS THIS SO SLOW?
unit_test_Riemann_curvature(seed=0)
unit_test_Ricci_curvature(seed=0)
unit_test_scalar_curvature(seed=0)
unit_test_sectional_curvature(seed=0)

unit_test_Classification()
