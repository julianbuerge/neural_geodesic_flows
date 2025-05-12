"""
General methods used for testing.

Each is passed a function to be tested.
"""

import jax
import jax.numpy as jnp

#each unit test should call this before the actual testing methods are called
def printheading(unit_name):
    print(f"\n############### unit test of {unit_name} ###############\n")

#tests if a given function has return values of the expected shape
def test_function_dimensionality(func, in_shapes):

    #generate inputs with the correct shapes
    inputs = [jnp.zeros(shape) for shape in in_shapes]

    #call the function with the generated inputs
    output = func(*inputs)

    out_shape = output.shape

    print("Dimensionality test\n")
    print(f"input shapes {in_shapes}\nobtained output shape {out_shape}\n")

#test if a given function has the same return value as a given correct function
def test_function_evaluation(func, correct_func, in_shapes, seed = 0):

    #generate random inputs with the correct shapes
    key = jax.random.PRNGKey(seed)
    inputs = [jax.random.normal(key, shape) for shape in in_shapes]

    #evaluate both functions with the generated inputs
    y = func(*inputs)
    y_correct = correct_func(*inputs)

    mae = jnp.mean(jnp.abs(y_correct - y))

    print("Evaluation test on random inputs\n")
    print(f"correct output\n{y_correct}")
    print(f"\nobtained output\n{y}")
    print(f"\nmeans absolute error {mae}\n")

#see the evaluation of a function (to perhaps see if it's in the expected range or the like)
def print_function_evaluation(func, in_shapes, seed = 0):

    #generate random inputs with the correct shapes
    key = jax.random.PRNGKey(seed)
    inputs = [jax.random.normal(key, shape) for shape in in_shapes]

    #evaluate the function with the generated inputs
    y = func(*inputs)

    print("Evaluation test on random inputs\n")
    print(f"obtained output\n{y}")


#test func for symmetric and positive definiteness
def test_metric_evaluation(func, in_size, seed = 0):

    #generate random input for testing
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (in_size,))

    #compute the metric tensor
    g = func(x)

    #check symmetry by calculating the mean absolute symmetry error
    symmetry_error = jnp.mean(jnp.abs(g - g.T))

    #calculate eigenvalues to check positive definiteness
    eigenvalues, _ = jax.scipy.linalg.eigh(g)  # Use eigh for symmetric matrices

    #check if all eigenvalues are positive for positive definiteness
    posdef = jnp.all(eigenvalues > 0)

    #check if symmetric, g = g.T
    symmetry_error = jnp.mean(jnp.abs((g - g.T)))

    print("SPD test:")
    print(f"- mean absolute symmetry error {symmetry_error}")
    if posdef:
        print(f"- Eigenvalues\n  {eigenvalues} > 0\n  => positive definite")
    else:
        print(f"- Eigenvalues\n  {eigenvalues}\n  => not positive definite!")
