"""
Collection of equinox.Module neural networks
intended for passing as psi (chart), phi (parametrization), or g (metric) in class TangentBundle instances.

Each neural network satifies:
-- It has exactly two arguments which are a dictionary called "arguments" and an optional argument key = jax.random.PRNGKey(0)
-- It has a member variable called "arguments" (dict) which saves all the "arguments"
-- It has a member variable "classname" (str)

Required inputs/ouputs:
x Networks for psi must have input = array of shape (2*dim_dataspace,) and output = array of shape (2*dim_M,)
x Networks for g must have input = array of shape (dim_M,) and output = array of shape (dim_M,dim_M) being a SPD matrix
x The network for phi have input = array of shape (2*dim_M,) and output = array of shape (dim_dataspace)
"""

import jax
import jax.numpy as jnp

import equinox as eqx

class identity_diffeomorphism(eqx.Module):
    #hardcoded identity map. Can be used for psi, phi

    arguments : dict
    classname: str

    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "identity_diffeomorphism"

    def __call__(self, y):

        return y

class NN_diffeomorphism(eqx.Module):
    #Basic neural network that can be used as a default for psi, phi

    layers: list

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), out_size (int), hidden_sizes (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'out_size', 'hidden_sizes']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        keys = jax.random.split(key, len(arguments['hidden_sizes']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_sizes = [arguments['in_size']] + arguments['hidden_sizes'] + [arguments['out_size']]
        self.layers = [eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i]) for i in range(len(layer_sizes) - 1)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_diffeomorphism"

    def __call__(self, y):

        for layer in self.layers[:-1]:

            y = jax.nn.tanh(layer(y))

        y = self.layers[-1](y)

        return y

class NN_diffeomorphism_for_chart(eqx.Module):
    #Basic neural network that can be used as a default for psi.
    #It has a tanh activation in the last layer, to render chart points initially in [-1,1]^2m.

    layers: list

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), out_size (int), hidden_sizes (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'out_size', 'hidden_sizes']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        keys = jax.random.split(key, len(arguments['hidden_sizes']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_sizes = [arguments['in_size']] + arguments['hidden_sizes'] + [arguments['out_size']]
        self.layers = [eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i]) for i in range(len(layer_sizes) - 1)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_diffeomorphism_for_chart"

    def __call__(self, y):

        for layer in self.layers:

            y = jax.nn.tanh(layer(y))

        return y

class NN_split_diffeomorphism(eqx.Module):
    #split neural network that can be used as a default for psi, phi.
    #basically we have two independent FCNs NN1 and NN2.
    #if y_in = (x,v) is a location and velocity we output y = (NN1(x), NN2(v))

    layers_x: list  #layers for processing x
    layers_v: list  #layers for processing v

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), out_size (int), hidden_sizes_x (list of ints), hidden_sizes_v (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'out_size', 'hidden_sizes_x', 'hidden_sizes_v']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        key_x, key_v = jax.random.split(key, 2)
        keys_x = jax.random.split(key_x, len(arguments['hidden_sizes_x']) + 1)
        keys_v = jax.random.split(key_v, len(arguments['hidden_sizes_v']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_x_sizes = [arguments['in_size']//2] + arguments['hidden_sizes_x'] + [arguments['out_size']//2]
        self.layers_x = [eqx.nn.Linear(layer_x_sizes[i], layer_x_sizes[i + 1], key=keys_x[i]) for i in range(len(layer_x_sizes) - 1)]

        layer_v_sizes = [arguments['in_size']//2] + arguments['hidden_sizes_v'] + [arguments['out_size']//2]
        self.layers_v = [eqx.nn.Linear(layer_v_sizes[i], layer_v_sizes[i + 1], key=keys_v[i]) for i in range(len(layer_v_sizes) - 1)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_split_diffeomorphism"

    def __call__(self, y):
        #expect y of size (1,2dim) or (2dim,)
        if y.ndim == 1:
            dim = y.shape[0] // 2
        elif y.ndim == 2:
            dim = y.shape[1] // 2
        else:
            raise ValueError("Input shape not compatible. Expected shape (2*dim,) or (1, 2*dim).")

        x = y[0:dim]
        v = y[dim:2*dim]

        # Pass x_in through NN1
        for layer in self.layers_x[:-1]:
            x = layer(x)
            x = jax.nn.tanh(x)
        x = self.layers_x[-1](x)

        # Pass v_in through NN2
        for layer in self.layers_v[:-1]:
            v = layer(v)
            v = jax.nn.tanh(v)
        v = self.layers_v[-1](v)

        y = jnp.concatenate([x, v], axis=-1)

        return y

class NN_linear_split_diffeomorphism(eqx.Module):
    #split neural network that can be used as a default for psi, phi.
    #basically we have two independent FCNs NN1 and NN2 where NN2 is one linear layer (no activation)
    #if y_in = (x,v) is a location and velocity we output y = (NN1(x), NN2(v))

    layers_x: list  #layers for processing x
    layer_v : list #will actually just be one element

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), out_size (int), hidden_sizes_x (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'out_size', 'hidden_sizes_x']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        key_x, key_v = jax.random.split(key, 2)
        keys_x = jax.random.split(key_x, len(arguments['hidden_sizes_x']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_x_sizes = [arguments['in_size']//2] + arguments['hidden_sizes_x'] + [arguments['out_size']//2]
        self.layers_x = [eqx.nn.Linear(layer_x_sizes[i], layer_x_sizes[i + 1], key=keys_x[i]) for i in range(len(layer_x_sizes) - 1)]

        self.layer_v = [eqx.nn.Linear(arguments['in_size']//2, arguments['out_size']//2, key=key_v)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_linear_split_diffeomorphism"

    def __call__(self, y):
        #expect y of size (1,2dim) or (2dim,)
        if y.ndim == 1:
            dim = y.shape[0] // 2
        elif y.ndim == 2:
            dim = y.shape[1] // 2
        else:
            raise ValueError("Input shape not compatible. Expected shape (2*dim,) or (1, 2*dim).")

        x = y[0:dim]
        v = y[dim:2*dim]

        # Pass x_in through NN1
        for layer in self.layers_x[:-1]:
            x = layer(x)
            x = jax.nn.tanh(x)
        x = self.layers_x[-1](x)

        #pass v through NN2 (one purely linear layer which is artificially in a list of size 1)
        v = self.layer_v[0](v)

        y = jnp.concatenate([x, v], axis=-1)

        return y

class NN_Jacobian_split_diffeomorphism(eqx.Module):
    #split neural network that can be used as a default for psi, phi.
    #basically we have a FCN NN for the location and its Jacobian for the velocities
    #if y_in = (x,v) is a location and velocity we output y = (NN(x), dNN_x(v))

    layers_x : list  #layers for processing x

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), out_size (int), hidden_sizes_x (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'out_size', 'hidden_sizes_x']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        keys_x = jax.random.split(key, len(arguments['hidden_sizes_x']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_x_sizes = [arguments['in_size']//2] + arguments['hidden_sizes_x'] + [arguments['out_size']//2]
        self.layers_x = [eqx.nn.Linear(layer_x_sizes[i], layer_x_sizes[i + 1], key=keys_x[i]) for i in range(len(layer_x_sizes) - 1)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_Jacobian_split_diffeomorphism"

    def nn(self, x):
        # Pass x_in through NN1
        for layer in self.layers_x[:-1]:
            x = layer(x)
            x = jax.nn.tanh(x)

        x = self.layers_x[-1](x)

        return x

    def __call__(self, y):
        #expect y of size (1,2dim) or (2dim,)
        if y.ndim == 1:
            dim = y.shape[0] // 2
        elif y.ndim == 2:
            dim = y.shape[1] // 2
        else:
            raise ValueError("Input shape not compatible. Expected shape (2*dim,) or (1, 2*dim).")

        x = y[0:dim]
        v = y[dim:2*dim]

        #get Jacobian at x
        dnn_x = jax.jacfwd(self.nn)(x)

        x = self.nn(x)
        v = dnn_x @ v

        y = jnp.concatenate([x, v], axis=-1)

        return y

class NN_Jacobian_split_diffeomorphism_for_chart(eqx.Module):
    #split neural network that can be used as a default for psi.
    #It has a tanh activation in the last layer, to render chart points initially in [-1,1]^m x R^m.

    #basically we have a FCN NN for the location and its Jacobian for the velocities
    #if y_in = (x,v) is a location and velocity we output y = (NN(x), dNN_x(v))

    layers_x : list  #layers for processing x

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), out_size (int), hidden_sizes_x (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'out_size', 'hidden_sizes_x']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        keys_x = jax.random.split(key, len(arguments['hidden_sizes_x']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_x_sizes = [arguments['in_size']//2] + arguments['hidden_sizes_x'] + [arguments['out_size']//2]
        self.layers_x = [eqx.nn.Linear(layer_x_sizes[i], layer_x_sizes[i + 1], key=keys_x[i]) for i in range(len(layer_x_sizes) - 1)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_Jacobian_split_diffeomorphism_for_chart"

    def nn(self, x):
        # Pass x_in through NN1
        for layer in self.layers_x[:-1]:
            x = layer(x)
            x = jax.nn.tanh(x)

        x = self.layers_x[-1](x)
        x = jax.nn.tanh(x)

        return x

    def __call__(self, y):
        #expect y of size (1,2dim) or (2dim,)
        if y.ndim == 1:
            dim = y.shape[0] // 2
        elif y.ndim == 2:
            dim = y.shape[1] // 2
        else:
            raise ValueError("Input shape not compatible. Expected shape (2*dim,) or (1, 2*dim).")

        x = y[0:dim]
        v = y[dim:2*dim]

        #get Jacobian at x
        dnn_x = jax.jacfwd(self.nn)(x)

        x = self.nn(x)
        v = dnn_x @ v

        y = jnp.concatenate([x, v], axis=-1)

        return y


class NN_conv_diffeomorphism_for_chart(eqx.Module):
    #Basic neural network that can be used as a default for psi
    #the inputs are images with several channels, so of shape (channels, x_res, y_res) = dim_dataspace
    #the outputs are points in TM of shape (2*dim_M,)

    #we go through some convolutional layers to (channels_new,x_res_new,y_res_new)
    #which we flatten and pass through some linear layers that output a point in [-1,1]^2dim_M as the chart point

    conv_layers: list
    linear_layers : list

    arguments : dict
    classname: str


    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essential keys are provided
        required_keys = ['dim_dataspace','dim_M','in_channel_sizes', 'out_channel_sizes', 'kernel_sizes','stride_sizes','padding_sizes','linear_hidden_sizes']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")


        #initialize random keys
        keys = jax.random.split(key, len(arguments['kernel_sizes']) + len(arguments['linear_hidden_sizes']) + 1)
        conv_keys = keys[:len(arguments['kernel_sizes'])]
        lin_keys = keys[len(arguments['kernel_sizes']):]

        #construct layer sizes based on input, hidden, and output sizes
        in_channel_sizes = arguments['in_channel_sizes']
        out_channel_sizes = arguments['out_channel_sizes']
        kernel_sizes = arguments['kernel_sizes']
        stride_sizes = arguments['stride_sizes']
        padding_sizes = arguments['padding_sizes']

        self.conv_layers = [eqx.nn.Conv2d(in_channels = in_channel_sizes[i],
                                     out_channels = out_channel_sizes[i],
                                     kernel_size = kernel_sizes[i],
                                     stride = stride_sizes[i],
                                     padding = padding_sizes[i],key=conv_keys[i]) for i in range(len(kernel_sizes))]

        #create a dummy input and apply the conv layers for it to find out what the input size of the first lin layer should be
        dummy = jnp.zeros(arguments['dim_dataspace']) #expect dim_dataspace to be a tuple of (channels, x_res, y_res)
        for layer in self.conv_layers:
            dummy = layer(dummy)

        print(f"NN_conv_diffeomorphism_for_chart initialized\nwhere conv image reduction will produce a {dummy.shape} image")

        dummy = jnp.ravel(dummy)

        linear_layer_sizes = [dummy.shape[0]] + arguments['linear_hidden_sizes'] + [2*arguments['dim_M']]
        self.linear_layers = [eqx.nn.Linear(linear_layer_sizes[i], linear_layer_sizes[i + 1], key=lin_keys[i]) for i in range(len(linear_layer_sizes) - 1)]


        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_conv_diffeomorphism_for_chart"

    #expect input of shape (channels, x res, y res)
    def __call__(self, y):

        #apply the conv layers to the input image to get a (2, x_res_new, y_res_new) shaped image
        for layer in self.conv_layers[:-1]:

            y = jax.nn.silu(layer(y))


        y = self.conv_layers[-1](y)
        y = jax.nn.silu(y)

        #reshape the image to a point
        y = jnp.ravel(y)

        #apply the linear layers to the point to get a TM point
        for layer in self.linear_layers:

            y = jax.nn.tanh(layer(y))

        #return the TM point
        return y

class NN_conv_diffeomorphism_for_parametrization(eqx.Module):
    #Basic neural network that can be used as a default for phi
    #the inputs are points in TM of shape (2*dim_M,)
    #the outputs are images with several channels, so of shape (channels, x_res, y_res) = dim_dataspace

    #we go through some linear layers that output a point of shape (intermediate_channels*intermediate_res^2,)
    #which we then reshape to an image (intermediate_channels, intermediate_res,intermediate_res)

    conv_layers: list
    linear_layers: list

    intermediate_channels : int
    intermediate_res : int

    arguments : dict
    classname: str


    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['dim_M',
                         'intermediate_channels',
                         'intermediate_resolution',
                         'linear_hidden_sizes',
                         'in_channel_sizes',
                         'out_channel_sizes',
                         'kernel_sizes',
                         'stride_sizes',
                         'padding_sizes']

        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #assign the resolution of the first images
        self.intermediate_channels = arguments['intermediate_channels']
        self.intermediate_res = arguments['intermediate_resolution']

        #initialize random keys
        keys = jax.random.split(key, len(arguments['kernel_sizes']) + len(arguments['linear_hidden_sizes']) + 1)
        conv_keys = keys[:len(arguments['kernel_sizes'])]
        lin_keys = keys[len(arguments['kernel_sizes']):]

        #construct linear layers
        linear_layer_sizes = [2*arguments['dim_M']] + arguments['linear_hidden_sizes'] + [self.intermediate_channels*self.intermediate_res**2]
        self.linear_layers = [eqx.nn.Linear(linear_layer_sizes[i], linear_layer_sizes[i + 1], key=lin_keys[i]) for i in range(len(linear_layer_sizes) - 1)]

        #construct conv layers
        in_channel_sizes = arguments['in_channel_sizes']
        out_channel_sizes = arguments['out_channel_sizes']
        kernel_sizes = arguments['kernel_sizes']
        stride_sizes = arguments['stride_sizes']
        padding_sizes = arguments['padding_sizes']

        self.conv_layers = [eqx.nn.Conv2d(in_channels = in_channel_sizes[i],
                                     out_channels = out_channel_sizes[i],
                                     kernel_size = kernel_sizes[i],
                                     stride = stride_sizes[i],
                                     padding = padding_sizes[i],key=conv_keys[i]) for i in range(len(kernel_sizes))]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_conv_diffeomorphism_for_parametrization"

    #expect input of shape (2*dim_M,)
    def __call__(self, y):

        #apply the linear layers to the input TM point to get a point of shape (intermediate_channels*intermediate_res^2)
        for layer in self.linear_layers[:-1]:

            y = jax.nn.silu(layer(y))

        y = self.linear_layers[-1](y)

        #reshape the point to an image
        y = y.reshape(self.intermediate_channels, self.intermediate_res, self.intermediate_res)

        #apply the upsampling and then conv layers to get the output image of shape (channels, x_res, y_res)
        for layer in self.conv_layers[:-1]:

            #upsample
            previous_channels = y.shape[0] #y has shape (channels, res, res)
            previous_res = y.shape[1]

            y = jax.image.resize(y, (previous_channels, previous_res*2, previous_res*2), method="bilinear")

            #apply convolution
            y = layer(y)
            y = jax.nn.silu(y)

        #upsample
        previous_channels = y.shape[0]
        previous_res = y.shape[1]

        y = jax.image.resize(y, (previous_channels, previous_res*2, previous_res*2), method="bilinear")

        #apply last layer
        y = self.conv_layers[-1](y)

        return y

class NN_pytorches_MNIST_encoder(eqx.Module):
    #copied architecture from pytorchs MNIST model.
    #to be used for psi.
    #this is not a diffeo as maxpool is not continuous (and relu is not differentiable)

    conv_layers : list
    linear_layers : list
    pool_layer : list

    arguments : dict
    classname : str

    def __init__(self, arguments, key = jax.random.PRNGKey(0)):


        #verify that essential keys are provided
        required_keys = []
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")


        #initialize random keys
        keys = jax.random.split(key, 4)

        self.conv_layers = [eqx.nn.Conv2d(in_channels = 1,out_channels = 32,kernel_size = 3,stride = 1,padding = 0,key=keys[0]),
                            eqx.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 0,key=keys[1])]

        self.pool_layer = [eqx.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)]

        self.linear_layers = [eqx.nn.Linear(9216, 128, key=keys[2]),
                              eqx.nn.Linear(128, 10, key=keys[3])]


        #assign remaining member variables
        self.arguments = arguments
        self.classname = "NN_pytorches_MNIST_encoder"


    #expect image of shape (channels, x_res, y_res)
    def __call__(self, y):

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


class identity_metric(eqx.Module):
    #the identity metric

    dim_M : int

    arguments : dict
    classname : str

    #dictionary "arguments" has to hold dim_M (int)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['dim_M']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required key '{dict_key}' in arguments")


        #assign member variables
        self.dim_M = arguments['dim_M']

        self.arguments = arguments
        self.classname = "identity_metric"

    #expect x of shape (dim_M,)
    def __call__(self, x):

        return jnp.eye(self.dim_M)

class NN_metric(eqx.Module):
    #Basic neural network that can be used as a default for g

    layers: list

    dim_M : int

    arguments: dict
    classname: str

    #dictionary "arguments" has to hold dim_M (int), hidden_sizes (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['dim_M', 'hidden_sizes']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required key '{dict_key}' in arguments")


        #initialize random keys
        keys = jax.random.split(key, len(arguments['hidden_sizes']) + 1)


        #create layers
        layer_sizes = [arguments['dim_M']] + arguments['hidden_sizes'] + [arguments['dim_M']*(arguments['dim_M']+1)//2]
        self.layers = [eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i]) for i in range(len(layer_sizes) - 1)]

        #assign remaining member variables
        self.dim_M = arguments['dim_M']

        self.arguments = arguments
        self.classname = "NN_metric"

    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.tanh(x)

        learnt_components = self.layers[-1](x)

        #populate a lower triangle (m by m) matrix L with m(m+1)/2 learnt components
        L = jnp.zeros((self.dim_M, self.dim_M))

        lower_triangle_indices = jnp.tril_indices(self.dim_M)

        L = L.at[lower_triangle_indices].set(learnt_components)

        #enforce positivity on the diagonal entries in a smooth and close to linear way
        L = L.at[jnp.diag_indices(self.dim_M)].set(jax.nn.softplus(jnp.diag(L)))

        #then LL^T is automatically symmetric positive definite
        #proof: y^T LL^T y = ||L^T y||^2 . It remains to show L^T is invertible. Note det(L^T) > 0 by construction. q.e.d.
        #CHECK SURJECTIVE
        g = L @ L.T

        return g

class NN_metric_regularized(eqx.Module):
    #Basic neural network that can be used as a default for g
    #It uses tanh activation in the last layer of L and returns g as a perturbation of the identity g = id + LL^T.

    layers: list

    dim_M : int

    arguments: dict
    classname: str

    #dictionary "arguments" has to hold dim_M (int), hidden_sizes (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['dim_M', 'hidden_sizes']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required key '{dict_key}' in arguments")


        #initialize random keys
        keys = jax.random.split(key, len(arguments['hidden_sizes']) + 1)


        #create layers
        layer_sizes = [arguments['dim_M']] + arguments['hidden_sizes'] + [arguments['dim_M']*(arguments['dim_M']+1)//2]
        self.layers = [eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i]) for i in range(len(layer_sizes) - 1)]

        #assign remaining member variables
        self.dim_M = arguments['dim_M']

        self.arguments = arguments
        self.classname = "NN_metric_regularized"

    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.tanh(x)

        learnt_components = self.layers[-1](x)
        learnt_components = jax.nn.tanh(learnt_components)

        #populate a lower triangle (m by m) matrix L with m(m+1)/2 learnt components
        L = jnp.zeros((self.dim_M, self.dim_M))

        lower_triangle_indices = jnp.tril_indices(self.dim_M)

        L = L.at[lower_triangle_indices].set(learnt_components)

        #enforce positivity on the diagonal entries in a smooth and close to linear way
        L = L.at[jnp.diag_indices(self.dim_M)].set(jax.nn.softplus(jnp.diag(L)))

        #then LL^T is automatically symmetric positive definite
        #proof: y^T LL^T y = ||L^T y||^2 . It remains to show L^T is invertible. Note det(L^T) > 0 by construction. q.e.d.
        g = L @ L.T

        #for stability add the identity matrix (sum of pos def is again pos def)
        g = jnp.eye(self.dim_M) + g

        return g
