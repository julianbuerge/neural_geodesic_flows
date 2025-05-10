"""
Unit tests for template_psi_phi_g_functions
"""

import jax
import jax.numpy as jnp

from tests.utils import (
    printheading,
    test_function_dimensionality,
    test_function_evaluation,
    test_metric_evaluation,
)

from core.models import TangentBundle

from core.template_psi_phi_g_functions_analytical import (
    two_body_Jacobi_metric,
)

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
    NN_pytorches_MNIST_encoder,
    identity_metric,
    NN_metric,
    NN_metric_regularized,
)

def unit_test_identity_diffeomorphism(seed = 0):

    dim_in = 10
    dim_out = 10

    arguments = {}

    nn = identity_diffeomorphism(arguments)

    id = lambda x : x

    printheading(unit_name=f"identity_diffeomorphism.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])

    test_function_evaluation(func = nn, correct_func = id, in_shapes = [(dim_in,)], seed = seed)

def unit_test_NN_diffeomorphism():

    dim_in = 10
    dim_out = 144

    arguments = {'in_size':dim_in,
                 'out_size':dim_out,
                 'hidden_sizes':[32,32,32]}

    nn = NN_diffeomorphism(arguments)

    printheading(unit_name=f"NN_diffeomorphism.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])

def unit_test_NN_diffeomorphism_for_chart():

    dim_in = 16
    dim_out = 12

    arguments = {'in_size':dim_in,
                 'out_size':dim_out,
                 'hidden_sizes':[32,32,32]}

    nn = NN_diffeomorphism_for_chart(arguments)

    printheading(unit_name=f"NN_diffeomorphism_for_chart.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])

def unit_test_NN_split_diffeomorphism():

    dim_in = 16
    dim_out = 12

    arguments = {'in_size':dim_in,
                 'out_size':dim_out,
                 'hidden_sizes_x':[32,32,32],
                 'hidden_sizes_v':[24,24]}

    nn = NN_split_diffeomorphism(arguments)

    printheading(unit_name=f"NN_split_diffeomorphism.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])

def unit_test_NN_linear_split_diffeomorphism():

    dim_in = 16
    dim_out = 12

    arguments = {'in_size':dim_in,
                 'out_size':dim_out,
                 'hidden_sizes_x':[32,32,32]}

    nn = NN_linear_split_diffeomorphism(arguments)

    printheading(unit_name=f"NN_linear_split_diffeomorphism.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])

def unit_test_NN_Jacobian_split_diffeomorphism():

    dim_in = 16
    dim_out = 12

    arguments = {'in_size':dim_in,
                 'out_size':dim_out,
                 'hidden_sizes_x':[32,32,32]}

    nn = NN_Jacobian_split_diffeomorphism(arguments)

    printheading(unit_name=f"NN_Jacobian_split_diffeomorphism.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])

def unit_test_NN_Jacobian_split_diffeomorphism_for_chart():

    dim_in = 16
    dim_out = 12

    arguments = {'in_size':dim_in,
                 'out_size':dim_out,
                 'hidden_sizes_x':[32,32,32]}

    nn = NN_Jacobian_split_diffeomorphism_for_chart(arguments)

    printheading(unit_name=f"NN_Jacobian_split_diffeomorphism_for_chart.__call__ of type {dim_in}--->{dim_out}")

    test_function_dimensionality(func = nn, in_shapes = [(dim_in,)])


def unit_test_NN_conv_diffeomorphism_for_chart():

    channels_data = 2
    x_res_data = 128
    y_res_data = 128
    dim_M = 40

    arguments = {'dim_dataspace':(channels_data,x_res_data,y_res_data),
                 'dim_M':dim_M,
                 'in_channel_sizes':[channels_data,4,8,16],
                 'out_channel_sizes':[4,8,16,32],
                 'kernel_sizes':[3,3,3,3],
                 'stride_sizes':[2,2,2,2],
                 'padding_sizes':[1,1,1,1],
                 'linear_hidden_sizes':[]}

    nn = NN_conv_diffeomorphism_for_chart(arguments)

    printheading(unit_name=f"NN_conv_diffeomorphism_for_chart.__call__ of type {channels_data}x{x_res_data}x{y_res_data} ---> {2*dim_M}")

    test_function_dimensionality(func = nn, in_shapes = [(channels_data,x_res_data,y_res_data)])

def unit_test_NN_conv_diffeomorphism_for_parametrization():

    x_res_data = 128
    y_res_data = 128
    dim_M = 40

    arguments = {'dim_M':40,
                 'intermediate_channels':32,
                 'intermediate_resolution':8,
                 'linear_hidden_sizes':[],
                 'in_channel_sizes':[32,16,8,4],
                 'out_channel_sizes':[16,8,4,2], #double every layer
                 'kernel_sizes':[3,3,3,3], #always 3 or 5
                 'stride_sizes':[1,1,1,1], #always 2
                 'padding_sizes':[1,1,1,1]} #no padding

    last_out_channel = arguments['out_channel_sizes'][-1]

    nn = NN_conv_diffeomorphism_for_parametrization(arguments)

    printheading(unit_name=f"NN_conv_diffeomorphism_for_parametrization.__call__ of type {2*dim_M} ---> {last_out_channel}x{x_res_data}x{y_res_data}")

    test_function_dimensionality(func = nn, in_shapes = [(2*dim_M,)])

def unit_test_NN_pytorches_MNIST_encoder():

    arguments = {}

    nn = NN_pytorches_MNIST_encoder(arguments)

    printheading(unit_name=f"NN_pytorches_MNIST_encoder.__call__")

    test_function_dimensionality(func = nn, in_shapes = [(1,28,28)])


def unit_test_identity_metric(seed=0):

    m = 6

    id = lambda x : jnp.eye(m)

    id_metric = identity_metric({'dim_M' : m})

    printheading(unit_name="identity_metric")

    test_function_dimensionality(func = id_metric, in_shapes = [(m,)])

    test_function_evaluation(func = id_metric,
                                correct_func = id, in_shapes = [(m,)],
                                    seed = seed)

def unit_test_NN_metric(seed=0):

    dim_M = 12
    arguments = {'dim_M':dim_M,
                 'hidden_sizes':[32,32,32]}

    nn = NN_metric(arguments)



    printheading(unit_name=f"NN_metric.__call__ for {dim_M}d manifold")

    test_function_dimensionality(func = nn, in_shapes = [(dim_M,)])

    test_metric_evaluation(func = nn, in_size = dim_M, seed = seed)

def unit_test_NN_metric_regularized(seed=0):

    dim_M = 12
    arguments = {'dim_M':dim_M,
                 'hidden_sizes':[32,32,32]}

    nn = NN_metric(arguments)

    printheading(unit_name=f"NN_metric_regularized.__call__ for {dim_M}d manifold")

    test_function_dimensionality(func = nn, in_shapes = [(dim_M,)])

    test_metric_evaluation(func = nn, in_size = dim_M, seed = seed)

def unit_test_two_body_Jacobi_metric(seed=0):

    id = lambda x : x

    arguments = {'total_energy': -0.153}

    nn = two_body_Jacobi_metric(id, arguments)

    printheading(unit_name=f"two_body_Jacobi_metric.__call__ on physical state space")

    test_function_dimensionality(func = nn, in_shapes = [(4,)])

    test_metric_evaluation(func = nn, in_size = 4, seed = seed)


############################### Testing #####################################

unit_test_identity_diffeomorphism(seed=0)
unit_test_NN_diffeomorphism()
unit_test_NN_diffeomorphism_for_chart()
unit_test_NN_split_diffeomorphism()
unit_test_NN_linear_split_diffeomorphism()
unit_test_NN_Jacobian_split_diffeomorphism()
unit_test_NN_Jacobian_split_diffeomorphism_for_chart()

unit_test_NN_conv_diffeomorphism_for_chart()
unit_test_NN_conv_diffeomorphism_for_parametrization()
unit_test_NN_pytorches_MNIST_encoder()

unit_test_identity_metric(seed=0)
unit_test_NN_metric(seed=0)
unit_test_NN_metric_regularized(seed=0)
unit_test_two_body_Jacobi_metric(seed=0)
