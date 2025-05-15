"""
Since NGFs are not apriori designed for classification
and we want to do MNIST classification some very specialized modifications are needed.

They are defined here and include:
- NN template for psi and phi tailored for the data/datasets/MNIST_train dataset
- classification loss and error methods
- unit tests for the above that get executed when this file is called
"""

import jax
import jax.numpy as jnp

import equinox as eqx

from core.models import (
    TangentBundle
)

from core.template_psi_phi_g_functions_neural_networks import (
    NN_metric_regularized
)

from tests.utils import (
    printheading,
    print_function_evaluation,
    test_function_dimensionality,
    test_function_evaluation,
    test_metric_evaluation,
)

##################################### NN templates #####################################

class NN_MNIST_encoder(eqx.Module):
    #NN to be used for psi when doing MNIST classification.
    #This is not a diffeomorphism as maxpool is not continuous (and relu is not differentiable).

    #The architecture is almost as in pytorchs MNIST example, found at https://github.com/pytorch/examples/blob/main/mnist/main.py

    conv_layers : list
    linear_layers : list
    pool_layer : list

    arguments : dict
    classname : str

    def __init__(self, arguments, key = jax.random.PRNGKey(0)):


        #verify that essential keys are provided
        required_keys = ['out_size']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")


        #initialize random keys
        keys = jax.random.split(key, 4)

        self.conv_layers = [eqx.nn.Conv2d(in_channels = 1,out_channels = 32,kernel_size = 3,stride = 1,padding = 0,key=keys[0]),
                            eqx.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 0,key=keys[1])]

        self.pool_layer = [eqx.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)]

        self.linear_layers = [eqx.nn.Linear(9216, 128, key=keys[2]),
                              eqx.nn.Linear(128, arguments['out_size'], key=keys[3])]


        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_MNIST_encoder"

    #since we want to use this class as an entire model,
    #it has to have this method which our applications.utils.save_model method requires
    def get_high_level_parameters(self):

        return {}

    #expect image of shape (channels, x_res, y_res)
    def __call__(self, y, *args,):

        #the dataset MNIST_train & MNIST_test are saved as int8 in [0,255] to decrease size
        y = jnp.asarray(y, dtype=jnp.float32) / 255

        #apply the conv layers to the input image to get a (channels_new, x_res_new, y_res_new) shaped image
        for layer in self.conv_layers:

            y = jax.nn.relu(layer(y))

        #do max pooling
        y = self.pool_layer[0](y)

        #reshape the image to a point
        y = jnp.ravel(y)

        #apply the linear layers to the point to get a TM-shaped point
        for layer in self.linear_layers[:-1]:

            y = jax.nn.relu(layer(y))

        y = self.linear_layers[-1](y)

        #apply a activation to get a class log probability (this is then the TM point)
        y = jax.nn.log_softmax(y)

        #return the TM point
        return y

class NN_MNIST_decoder(eqx.Module):
    #NN to be used for phi when doing MNIST classification.
    #This NN simply applies a log softmax, and returns the first dim M components,
    #such that a chart point in TM is turned into a collection of log probabilities,
    #and the ones from M are returned (requires dim M = amount of classes)

    #BE WARE: This "phi" is not a parametrization. ONLY use for classification.

    dim_M : int

    arguments : dict
    classname : str

    def __init__(self, arguments, key = jax.random.PRNGKey(0)):


        #verify that essential keys are provided
        required_keys = ['dim_M']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        self.dim_M = arguments['dim_M']

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_MNIST_decoder"


    #expect point in TM
    def __call__(self, z):

        #apply a activation to get class log probabilities
        p = jax.nn.log_softmax(z)

        #restrict to dim M
        p = p[0:self.dim_M]

        return p

##################################### loss functions #####################################

#expect data of shape (batch_size, mathematical dimension), (batch_size,), (batch_size)
def classification_loss(model, inputs, targets, times):

    #vectorize the forward call which we expect to take in a single input
    forward = jax.vmap(model, in_axes = (0,0,None))

    #choose a integration resolution
    num_steps = 25

    #generate classifications, expect array of shape (many,amount of classes)
    log_classifications_probabilities = forward(inputs, times, num_steps)

    #cross entropy loss for classification
    cross_entropy_loss = -jnp.sum(targets * log_classifications_probabilities, axis=1)
    mean = jnp.mean(cross_entropy_loss)

    #loss
    return mean

#expect data of shape (batch_size, mathematical dimension), (batch_size,), (batch_size)
def classification_error(model, inputs, targets, times):

    #vectorize the forward call which we expect to take in a single input
    forward = jax.vmap(model, in_axes = (0,0,None))

    #choose a integration resolution
    num_steps = 25

    #generate classifications, expect array of shape (many,amount of classes)
    log_classification_probabilities = forward(inputs, times, num_steps)

    #assigned classes shape (many,) (log is monotonously increasing)
    predicted_classes = jnp.argmax(log_classification_probabilities, axis=1)
    correct_classes = jnp.argmax(targets, axis = 1)

    errors = (predicted_classes != correct_classes).astype(jnp.float32)  # 1 for incorrect, 0 for correct

    error_rate = jnp.mean(errors)

    #error as a percentage (between 0 and 1)
    return error_rate


##################################### unit tests #####################################

def unit_test_NN_MNIST_encoder():

    arguments = {'out_size':20}

    nn = NN_MNIST_encoder(arguments)

    printheading(unit_name=f"NN_MNIST_encoder.__call__ of type (1,28,28) ---> {arguments['out_size']}")

    test_function_dimensionality(func = nn, in_shapes = [(1,28,28)])

def unit_test_NN_MNIST_decoder():

    dim_M = 10

    arguments = {'dim_M':dim_M}

    nn = NN_MNIST_decoder(arguments)

    printheading(unit_name=f"NN_MNIST_decoder.__call__ of type {2*dim_M} ---> {dim_M}")

    test_function_dimensionality(func = nn, in_shapes = [(2*dim_M,)])

def unit_test_classif_loss(seed=0):

    psi_NN = NN_MNIST_encoder({'out_size':20})

    phi_NN = NN_MNIST_decoder({'dim_M':10})

    g_NN = NN_metric_regularized({'dim_M' : 10,
                      'hidden_sizes' : [16,16]})

    tangentbundle = TangentBundle(dim_dataspace = (1,28,28), dim_M = 10,
                                  psi = psi_NN, phi = phi_NN, g = g_NN)

    loss_dim = lambda inputs, targets, times : classification_loss(model = tangentbundle,
                                                            inputs = inputs,
                                                            targets = targets,
                                                            times = times)

    loss_eval = lambda inputs : classification_loss(model = tangentbundle,
                                        inputs = inputs,
                                        targets = jnp.ones((100,10)),
                                        times = jnp.ones((100,)))

    printheading(unit_name="classification_loss")

    test_function_dimensionality(func = loss_dim, in_shapes = [(100,1,28,28),(100,10),(100,)])

    print_function_evaluation(func = loss_eval, in_shapes = [(100,1,28,28)],seed=seed)

def unit_test_classif_error(seed=0):

    psi_NN = NN_MNIST_encoder({'out_size':20})

    phi_NN = NN_MNIST_decoder({'dim_M':10})

    g_NN = NN_metric_regularized({'dim_M' : 10,
                      'hidden_sizes' : [16,16]})

    tangentbundle = TangentBundle(dim_dataspace = (1,28,28), dim_M = 10,
                                  psi = psi_NN, phi = phi_NN, g = g_NN)

    loss_dim = lambda inputs, targets, times : classification_error(model = tangentbundle,
                                                            inputs = inputs,
                                                            targets = targets,
                                                            times = times)

    #the random inputs will have these 7 fixed targets. So the resulting error is also random.
    loss_eval = lambda inputs : classification_error(model = tangentbundle,
                                        inputs = inputs,
                                        targets = jnp.array([[0,1.0,0,0,0,0,0,0,0,0],
                                                             [0,0,0,0,0,0,0,1.0,0,0],
                                                             [0,0,0,1.0,0,0,0,0,0,0],
                                                             [1.0,0,0,0,0,0,0,0,0,0],
                                                             [0,0,0,0,0,0,0,0,0,1.0],
                                                             [0,1.0,0,0,0,0,0,0,0,0],
                                                             [0,0,0,0,0,1.0,0,0,0,0]]),
                                        times = jnp.ones((7,)))

    printheading(unit_name="classification_error")

    test_function_dimensionality(func = loss_dim, in_shapes = [(100,1,28,28),(100,7),(100,)])

    print_function_evaluation(func = loss_eval, in_shapes = [(7,1,28,28)],seed=seed)

#if this module is executed, perform the unit tests of the methods defined here
if __name__ == "__main__":

    unit_test_NN_MNIST_encoder()
    unit_test_NN_MNIST_decoder()

    unit_test_classif_loss(seed=0)
    unit_test_classif_error(seed=0)
