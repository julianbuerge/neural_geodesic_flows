"""
Module containing several classes and methods with which a multi chart atlas can work.

Overview:
- an atlas will be a tuple of Chart instances
- each Chart contains a CoordinateDomain instance and three functions: psi, phi, g.
- A CoordinateDomain consists of interior and boundary points, to allow some neat domain switching logic
    ---> all these are combined in this intended way in the main multi chart NGF model, the class TangetBundle_multi_chart_atlas

- create_coordinate_domains and create_atlas are methods meant to be used for learning, starting with a dataset
  and ending up with a tuple "atlas" that can be passed to an instance of TangetBundle_multi_chart_atlas
"""

import jax
import jax.numpy as jnp

import equinox as eqx

#patch of a manifold. We have interior and boundary points.
#the boundary points are meant to overlap with some other domain.
#each boundary points uniquely belong to the interior of another domain
#the index of which is saved in boundary_new_chart_ids
#(these ids must be managed whereever we use instances of this class)
class CoordinateDomain(eqx.Module):

    #centroid point of the domain
    centroid: jnp.ndarray

    #collection of interior points
    interior_points: jnp.ndarray

    #collection of boundary points, those overlap with the interior of other charts
    boundary_points: jnp.ndarray

    #each boundary point uniquely lays in the interior of another chart, this array holds the ids of these other charts
    boundary_new_chart_ids: jnp.ndarray

    def __init__(self, centroid, interior_points, boundary_points, boundary_new_chart_ids):

        self.centroid = centroid
        self.interior_points = interior_points
        self.boundary_points = boundary_points
        self.boundary_new_chart_ids = boundary_new_chart_ids

#The main NGF model TangetBundle_multi_chart_atlas has as its core a tuple "atlas"
#of Chart instances. These are all responsible for a certain coordinate domain
class Chart(eqx.Module):

    coordinate_domain : CoordinateDomain

    psi: callable
    phi: callable
    g: callable

    def __init__(self, coordinate_domain, psi , phi , g):

        self.coordinate_domain = coordinate_domain

        self.psi = psi
        self.phi = phi
        self.g = g

    def connection_coeffs(self, x):

        G = lambda x : self.g(x)

        partial_g = jax.jacfwd(G)(x)
        partial_g = jnp.transpose(partial_g, axes=(2, 0, 1))

        inverse_g = jnp.linalg.inv(self.g(x))

        term1 = jnp.einsum('ki,aib->kab', inverse_g, partial_g, optimize="optimal")  # 1st term: partial_b g_ia
        term2 = jnp.einsum('ki,bai->kab', inverse_g, partial_g, optimize="optimal")  # 2nd term: partial_a g_ib
        term3 = jnp.einsum('ki,iab->kab', inverse_g, partial_g, optimize="optimal")  # 3rd term: partial_i g_ab

        Gamma = 0.5 * (term1 + term2 - term3)

        return Gamma  #shape (m,m,m)

    def geodesic_ODE_function(self, z):

        m = z.shape[0]//2

        x = z[0:m]
        v = z[m:2*m]

        Gamma = self.connection_coeffs(x) #shape (m,m,m)

        dxbydt = v

        dvbydt = -jnp.einsum('kab, a, b -> k', Gamma, v, v, optimize="optimal") # -Gamma^k_{ab} v^a v^b

        return jnp.concatenate((dxbydt,dvbydt))

    #this is a single Runge Kutta 4 time step of the geodesic equation
    def exp_single_time_step(self, z, dt):

        k1 = self.geodesic_ODE_function(z)
        k2 = self.geodesic_ODE_function(z + 0.5*dt*k1)
        k3 = self.geodesic_ODE_function(z + 0.5*dt*k2)
        k4 = self.geodesic_ODE_function(z + dt*k3)

        z = z + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

        return z

    def exp(self, z, t, num_steps: int):

        dt = t/num_steps

        def step_function(z, _):
            z_next = self.exp_single_time_step(z, dt)
            return z_next, None

        z, _ = jax.lax.scan(step_function, z, None, length=num_steps)

        return z

    #same as exp above, but this version returns the whole geodesic trajectory
    def exp_return_trajectory(self, z, t, num_steps: int):

        #step function needed for the jax.lax.scan call, now returns each intermediate z value
        def step_function(z, _):
            z_next = self.exp_single_time_step(z, dt)
            return z_next, z_next  # Return z_next as both state and output for lax.scan to collect it

        #find dt based on the number of steps
        dt = t / num_steps

        #use jax.lax.scan to run the integration loop, obtaining the whole geodesic trajectory. the last point SHOULD correspond to exp of the same input
        _, geodesic_trajectory = jax.lax.scan(step_function, z, None, length=num_steps)

        #return the whole geodesic trajectory, shape (1 + num_steps, dim M)
        return jnp.vstack([z, geodesic_trajectory])


    #similar in purpose to exp_return_trajectory above, but here we take an initial y from the dataspace
    #and return the whole geodesic trajectory in the dataspace
    def get_geodesic(self, y, t, num_steps: int):

        z = self.psi(y)

        z_geo = self.exp_return_trajectory(z, t, num_steps)

        y_geo = jax.vmap(self.phi, in_axes = 0)(z_geo)

        #return the whole geodesic, shape (1 + num_steps, dim dataspace)
        return y_geo

    def __call__(self, y_in, t, num_steps : int):


        z_in = self.psi(y_in)

        z_out = self.exp(z_in, t, num_steps)

        y_out = self.phi(z_out)

        return y_out


#CURRENTLY HARDCODED FOR THE TWO SPHERE WITH TWO COORDIANTE DOMAINS: SLIGHTLY EXTENDED UPPER AND LOWER HEMISPHERES
#this method turns a manifold given as a collection of data points into
#a partition of clusters. It then extends the clusters to overlaping clusters.
#these are the domains. It returns these domains in the tuple format specified in the class Chart
def create_coordinate_domains(dataset, k, extension_degree):

    #returns tuple (or better different format?) of clusters, each point marked as interior, and one as the centroid
    def k_means(dataset, k):
        pass

    #increases a cluster with neighbouring points, marked as boundary
    def extend_cluster(cluster, extension_degree):
        pass

    #expect dataset with points (y1, y2, y3) and extract the y3 coordinate
    y3_vals = dataset[:, 2]


    #create boolean masks for the interior points
    mask_upper = y3_vals > 0.0
    mask_lower = y3_vals < 0.0

    upper_interior_points = dataset[mask_upper]
    lower_interior_points = dataset[mask_lower]


    #create boolean masks for the boundary points
    mask_boundary_upper = (y3_vals > -0.2) & (y3_vals <= 0.0)
    mask_boundary_lower = (y3_vals < 0.2) & (y3_vals >= 0.0)

    upper_boundary_points = dataset[mask_boundary_upper]
    lower_boundary_points = dataset[mask_boundary_lower]

    #assign the indices of the other
    upper_boundary_new_chart_ids = jnp.ones(upper_boundary_points.shape[0], dtype=int) * 1  # belong to the interior of chart 1
    lower_boundary_new_chart_ids = jnp.ones(lower_boundary_points.shape[0], dtype=int) * 0  # belong to the interior of chart 0

    #create centroids
    upper_centroid =  jnp.array([0.0,0.0,1.0])
    lower_centroid =  jnp.array([0.0,0.0,-1.0])


    #finally create the coordinate_domains of the correct structure
    upper_coordinate_domain = CoordinateDomain(upper_centroid,
                                               upper_interior_points,
                                               upper_boundary_points,
                                               upper_boundary_new_chart_ids)
    lower_coordinate_domain = CoordinateDomain(lower_centroid,
                                               lower_interior_points,
                                               lower_boundary_points,
                                               lower_boundary_new_chart_ids)

    return (upper_coordinate_domain, lower_coordinate_domain)


#given a tuple of coordinate domains
#assign a psi,phi,g function to each, based on some class initializer
#(which has to be an eqx.Module taking two arguments: a dictionary and a key)
#and then return a tuple of Chart instances:
#atlas = (chart_1, chart_2, ..., chart_k)
def create_atlas(domains: tuple,
                 psi_initializer: callable,
                 phi_initializer: callable,
                 g_initializer: callable,
                 psi_arguments: dict,
                 phi_arguments: dict,
                 g_arguments: dict,
                 key = jax.random.PRNGKey(0)):

    psi_key, phi_key, g_key = jax.random.split(key, 3)
    psi_keys = jax.random.split(psi_key, len(domains))
    phi_keys = jax.random.split(phi_key, len(domains))
    g_keys = jax.random.split(g_key, len(domains))

    atlas = ()

    for i, domain in enumerate(domains):

        psi = psi_initializer(psi_arguments, keys[i])
        phi = phi_initializer(phi_arguments, phi_keys[i])
        g = g_initializer(g_arguments, g_keys[i])

        chart = Chart(coordinate_domain = domain,
                      psi = psi,
                      phi = phi,
                      g = g)

        atlas = atlas + (chart,)

    return atlas




#NOT USED AT THE MOMENT, EXCEPT IN (ALSO NOT USED AND INCOMPLETE) TangetBundle_partition_of_unity_atlas
class Neural_partition_of_unity(eqx.Module):
    #neural network that takes an input x and outputs a smooth assignment of numbers p_1,...,p_k in [0,1]
    #such that p_1 + ... + p_k = 1

    layers: list

    arguments : dict
    classname: str

    #dictionary "arguments" has to hold in_size (int), amount_of_domains (int), hidden_sizes (list of ints)
    def __init__(self, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['in_size', 'amount_of_domains', 'hidden_sizes']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required argument: '{dict_key}'")

        #initialize random keys
        keys = jax.random.split(key, len(arguments['hidden_sizes']) + 1)

        #construct layer sizes based on input, hidden, and output sizes
        layer_sizes = [arguments['in_size']] + arguments['hidden_sizes'] + [arguments['amount_of_domains']]
        self.layers = [eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i]) for i in range(len(layer_sizes) - 1)]

        #assign remaining member variables
        self.arguments = arguments
        self.classname = "Neural_partition_of_unity"

    def mollifier(self, c):

        return jnp.where(c > 0, jnp.exp(-1/c), 0.0)

    def __call__(self, x):

        for layer in self.layers[:-1]:

            x = jax.nn.silu(layer(x))

        #array of shape (amount_of_charts,)
        c = self.layers[-1](x)

        #make them be in [0, 1]
        p = jax.vmap(self.mollifier)(c)

        #normalize such that the sum equals 1
        p = p / jnp.sum(p)

        return p
