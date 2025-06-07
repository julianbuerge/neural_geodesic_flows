"""
The classes defined here will later be moved to core.models

The class TangetBundle_multi_chart_atlas is the new version of TangetBundle and is
the main NGF model with a multi chart atlas. It's essence is a tuple "atlas" of
Chart instances. It's logic is mainly chart switching and calling geometric methods
from the currently active chart.
"""

import jax
import jax.numpy as jnp

import equinox as eqx

from experimental.atlas import (
    CoordinateDomain
)


class TangetBundle_multi_chart_atlas(eqx.Module):

    #tuple of Chart instances
    atlas : tuple

    amount_of_charts : int

    def __init__(self, atlas):

        self.atlas = atlas

        self.amount_of_charts = len(atlas)


    #THIS METHOD IS NOT IN USE, WE ALWAYS USE THE MORE GENERAL ONE BELOW
    #apply a function in (or accossiated with) each chart (through the passed tuple of functions)
    #to the inputs (which are a tuple of all required arguments)
    def apply_function_same_input_in_all_charts(self, functions, chart_id, inputs):

        # Create a tuple of callables, one for each chart function, to be used with jax.lax.switch.
        # Each callable must take no arguments and return the same pytree structure.
        #
        # We use the pattern (lambda f=f: lambda: f(*inputs))() to safely capture each function `f`:
        # - The outer lambda with `f=f` binds `f` by value (not by reference), avoiding Python closure pitfalls.
        # - The inner lambda is the actual no-arg function JAX expects.
        # - The outer lambda is immediately invoked, returning the correct inner lambda.
        #
        # This ensures each branch is independent, correctly scoped, and not prematurely evaluated.

        branches = tuple( (lambda f=f: lambda: f(*inputs))() for f in functions)

        return jax.lax.switch(chart_id, branches)


    #apply a function in (or accossiated with) each chart (through the passed tuple of functions)
    #to the inputs (which are a tuple of tuples, the latter being all required arguments)
    def apply_function(self, functions, inputs, chart_id):

        # Applies the i-th function in `functions` to the i-th tuple of inputs in `inputs`,
        # but only evaluates and returns the result from the function at index `chart_id`.
        #
        # Args:
        #   functions: A tuple of callables, one for each chart, all with compatible return shapes.
        #   inputs: A tuple of tuples, where each inner tuple contains the arguments for the corresponding function.
        #   chart_id: An integer index specifying which function's output to return.
        #
        # Internally, this uses `jax.lax.switch`, which requires that each branch (i.e., function applied to inputs)
        # has exactly the same output shape and dtype. The outer generator builds each switch branch using
        # a lambda closure: (lambda f=f, inp=inp: lambda: f(*inp))(). Detailed explanation in the method above
        #
        # Only the selected function's branch (based on `chart_id`) will be JIT-evaluated (others are lazy).


        branches = tuple((lambda f=f, inp=inp: lambda: f(*inp))() for f, inp in zip(functions, inputs))

        return jax.lax.switch(chart_id, branches)

    #the coordinate domains are saved in each Chart instance in the atlas in the data space
    #this method encodes them all into the latent space
    #it also pads them with the last element such that all corresponding arrays are the same shape across the atlas
    #(required for jax.lax.scan in apply_function)
    def encode_coordinate_domains(self, atlas):

        encoded_domains = ()

        #to find out the lenght of the largest interior_points array
        max_interior = 0
        #to find out the lenght of the largest boundary_points array (and boundary_new_chart_ids array)
        max_boundary = 0

        ### encode all the domains from data to latent space ###
        for chart in atlas:

            encoder = jax.vmap(chart.psi)

            #obtain the coordinate domain mapped to the chart
            encoded_centroid = chart.psi(chart.coordinate_domain.centroid)
            encoded_interior_points = encoder(chart.coordinate_domain.interior_points)
            encoded_boundary_points = encoder(chart.coordinate_domain.boundary_points)

            encoded_domain = CoordinateDomain(encoded_centroid,
                                              encoded_interior_points,
                                              encoded_boundary_points,
                                              chart.coordinate_domain.boundary_new_chart_ids)

            max_interior = max(max_interior, encoded_interior_points.shape[0])
            max_boundary = max(max_boundary, encoded_boundary_points.shape[0])

            encoded_domains = encoded_domains + (encoded_domain,)

        ### pad them so that arrays are the same shape across different domains ###
        def pad_to_length(array, target_len):

            pad_length = target_len - array.shape[0]

            if pad_length <= 0:
                return array

            padding = jnp.repeat(array[-1:], pad_length, axis=0)

            return jnp.concatenate([array, padding], axis=0)


        padded_encoded_domains = ()

        for domain in encoded_domains:

            padded_interior_points = pad_to_length(domain.interior_points, max_interior)
            padded_boundary_points = pad_to_length(domain.boundary_points, max_boundary)
            padded_boundary_new_chart_ids = pad_to_length(domain.boundary_new_chart_ids, max_boundary)

            padded_domain = CoordinateDomain(domain.centroid,
                                             padded_interior_points,
                                             padded_boundary_points,
                                             padded_boundary_new_chart_ids)

            padded_encoded_domains = padded_encoded_domains + (padded_domain,)

        return padded_encoded_domains

    #for an input in the dataspace find out which coordinate domain it belongs to and return its chart_id
    def determine_chart(self, y):

        #this is currently implemented with the assumption that we are operating on a 2n dimensional data tangent bundle
        #with coordinate domains on the non tangent part only.
        def distance_to_centroid(centroid, y):

            n = y.shape[0]//2

            return jnp.sum((y[:n] - centroid) ** 2)

        #find the distance to each coordinate domains centroid
        distances_all = jnp.stack([distance_to_centroid(chart.coordinate_domain.centroid, y) for chart in self.atlas])

        #the smallest one is the chart we belong to
        chart_id = jnp.argmin(distances_all)  # shape (), index of the smallest distance

        return chart_id


    #test if z should stay in the chart chart_id or if we need to switch to another
    #if so, return the (new chart_id, z) in the *new* chart
    #if not, return the same (chart_id, z)
    def update_chart(self, state, encoded_domains):

        chart_id, z = state

        #functions to calculate the distances
        def distance(z_other, z):

            m = z.shape[0]//2

            return jnp.sum( (z_other[0:m] - z[0:m])**2 )

        distances = jax.vmap(distance, in_axes = (0, None))

        #distances to all interior points, shape (many,)
        dist_to_interior_points = self.apply_function(functions = tuple(distances for i in range(self.amount_of_charts)),
                                                      inputs = tuple((domain.interior_points,z) for domain in encoded_domains),
                                                      chart_id = chart_id)

        #distance to all boundary points, shape (different many,)
        dist_to_boundary_points = self.apply_function(functions = tuple(distances for i in range(self.amount_of_charts)),
                                                      inputs = tuple((domain.boundary_points,z) for domain in encoded_domains),
                                                      chart_id = chart_id)

        #distance to all boundary points, shape (different many,)
        boundary_new_chart_ids = self.apply_function(functions = tuple(lambda x : x for i in range(self.amount_of_charts)),
                                                     inputs = tuple((domain.boundary_new_chart_ids,) for domain in encoded_domains),
                                                     chart_id = chart_id)

        dist_to_interior = jnp.min(dist_to_interior_points)
        dist_to_boundary = jnp.min(dist_to_boundary_points)

        switch_required = dist_to_boundary < dist_to_interior

        #if closest to a boundary point, switch to the chart that this boundary point is in the interior of
        def switch():

            index_of_closest_point = jnp.argmin(dist_to_boundary_points)

            new_chart_id = jax.lax.dynamic_index_in_dim(boundary_new_chart_ids, index_of_closest_point, axis=0, keepdims=False)
            new_chart_id = jnp.asarray(new_chart_id, dtype=jnp.int32)

            #decode from the old chart to the data space
            y = self.apply_function(functions = tuple(chart.phi for chart in self.atlas),
                                    inputs = tuple((z,) for i in range(self.amount_of_charts)),
                                    chart_id = chart_id)

            #encode into the new chart
            new_z = self.apply_function(functions = tuple(chart.psi for chart in self.atlas),
                                        inputs = tuple((y,) for i in range(self.amount_of_charts)),
                                        chart_id = new_chart_id)

            return (new_chart_id, new_z)

        new_state = jax.lax.cond(switch_required, switch, lambda : (chart_id, z))

        return new_state

    #encoder
    def psi(self, y):

        chart_id = self.determine_chart(y)

        z = self.apply_function(functions = tuple(chart.psi for chart in self.atlas),
                                inputs = tuple((y,) for i in range(self.amount_of_charts)),
                                chart_id = chart_id)

        return (chart_id, z)

    #decoder
    def phi(self, state):

        chart_id, z = state

        y = self.apply_function(functions = tuple(chart.phi for chart in self.atlas),
                                inputs = tuple((z,) for i in range(self.amount_of_charts)),
                                chart_id = chart_id)

        return y

    #metric
    def g(self, state):

        chart_id, z = state

        m = z.shape[0]//2

        x = z[0:m]

        G = self.apply_function_better_but_same_input_for_all(functions = tuple(chart.g for chart in self.atlas),
                                                              inputs = tuple((x,) for i in range(self.amount_of_charts)),
                                                              chart_id = chart_id)

        return G

    #geodesic evolution, state = (chart_id, z)
    def exp_return_trajectory(self, state, t, num_steps: int):

        dt = t / num_steps

        encoded_domains = self.encode_coordinate_domains(self.atlas)

        def step_function(state, _):

            #retrieve current chart and latent value
            chart_id, z = state

            z_next = self.apply_function(functions = tuple(chart.exp_single_time_step for chart in self.atlas),
                                         inputs = tuple((z,dt) for i in range(self.amount_of_charts)),
                                         chart_id = chart_id)

            #possibly switch chart
            next_state = (chart_id, z_next)

            next_state = self.update_chart(next_state, encoded_domains)

            return next_state, next_state

        initial_state = state

        #obtain tuple of two arrays, one is the chart_ids shape (num_steps,), the other the trajectory shape (num_steps, math dim)
        _, geodesic_evolution = jax.lax.scan(step_function, initial_state, None, length=num_steps)

        chart_ids_evolution, zs_evolution = geodesic_evolution

        #obtain the two arrays from the initial state
        chart_id_initial, z_initial = initial_state

        chart_id_initial = jnp.expand_dims(chart_id_initial, axis=0)
        z_initial = jnp.expand_dims(z_initial, axis=0)


        #prepend the initial state to obtain the full geodesic (initial & evolution)
        chart_ids_geodesic = jnp.concatenate([chart_id_initial,
                                              chart_ids_evolution],
                                              axis=0)

        zs_geodesic = jnp.concatenate([z_initial,
                                       zs_evolution], axis=0)

        #tuple with arrays of shape (num_steps + 1,) & (num_steps + 1, math dim)
        geodesic_states = (chart_ids_geodesic, zs_geodesic)

        return geodesic_states


#ONLY PARTIALLY IMPLEMENTED AND NOT USED AT THE MOMENT. I THINK IT WOULDN'T WORK EITHER
class TangetBundle_partition_of_unity_atlas(eqx.Module):

    #partition of unity function defined on the latent space reduced to M, so inputs are x in M, where z=(x,v) is a latent point in TM
    partition_of_unity : callable

    #we use one single encoder
    psi : callable

    #tuple of decoders and metrics
    decoders : tuple
    metrics : tuple

    dim_M : int
    amount_of_charts : int

    def __init__(self, dim_M, partition_of_unity, encoder, decoders, metrics):

        self.partition_of_unity = partition_of_unity

        self.psi = encoder

        self.decoders = decoders
        self.metrics = metrics

        self.dim_M = dim_M
        self.amount_of_charts = len(decoders)


    #decoder
    def phi(self, z):

        x = z[0:self.dim_M]

        #yields shape (amount_of_charts,)
        p = jax.vmap(self.partition_of_unity)(x)

        #yields shape (amount_of_charts, math dim)
        y_all = jnp.stack([decoder(z) for decoder in self.decoders])

        y = jnp.dot(p, y_all)

        return y

    #metric
    def g(self, x):

        p = self.partition_of_unity(x)

        G_all = jnp.stack([metric(x) for metric in self.metrics])

        G = jnp.dot(p, G_all)

        return G

    ######## from here on everything is the same as in TangentBundle_single_chart_atlas ########

    #evaluate the scalarproduct at x of u and v on M being g_x(u,v)
    def scalarproduct(self, x, u, v):

        return u @ self.g(x) @ v



    #at x in U return Gamma(x) of shape (m,m,m)
    #it is important to calculate the derivate/inverse afresh each call as g might have changed by some learning.
    def connection_coeffs(self, x):
        m = self.dim_M

        #compute partial derivatives of g w.r.t x. shape (m,m,m) and after transposition partial_i g_ab stored at [i,a,b]
        G = lambda x : self.g(x)

        partial_g = jax.jacfwd(G)(x)
        partial_g = jnp.transpose(partial_g, axes=(2, 0, 1))

        #compute inverse of g(x) once
        inverse_g = jnp.linalg.inv(self.g(x))

        #formula (summation over i) Gamma^k_ab = 1/2 g^ki ( partial_a g_ib + partial_b g_ai - partial_i g_ab)
        #compute each term of shape (m,m,m)
        term1 = jnp.einsum('ki,aib->kab', inverse_g, partial_g, optimize="optimal")  # 1st term: partial_b g_ia
        term2 = jnp.einsum('ki,bai->kab', inverse_g, partial_g, optimize="optimal")  # 2nd term: partial_a g_ib
        term3 = jnp.einsum('ki,iab->kab', inverse_g, partial_g, optimize="optimal")  # 3rd term: partial_i g_ab

        #combine terms to get Gamma^k_ab
        Gamma = 0.5 * (term1 + term2 - term3)

        return Gamma  #shape (m,m,m)



    #function f of the geodesic ODE in form dz/dt = f(z)
    def geodesic_ODE_function(self, z):
        m = self.dim_M

        x = z[0:m]
        v = z[m:2*m]

        Gamma = self.connection_coeffs(x) #shape (m,m,m)

        dxbydt = v

        dvbydt = -jnp.einsum('kab, a, b -> k', Gamma, v, v, optimize="optimal") # -Gamma^k_{ab} v^a v^b

        return jnp.concatenate((dxbydt,dvbydt))


    #geodesic solver:
    #starting at z=(x_initial,v_initial) in TU return exp(x_initial,v_initial,t_final) in TU,
    #that is, the point c(t_final) on the unique geodesic c(t) defined through c(0) = x_initial, dc/dt(0) = v_initial
    def exp(self, z, t, num_steps: int):

        #find dt based on the amount of steps
        dt = t/num_steps

        #perform a single Runge Kutta 4 time step
        def RK4_step(z, dt):
            k1 = self.geodesic_ODE_function(z)
            k2 = self.geodesic_ODE_function(z + 0.5*dt*k1)
            k3 = self.geodesic_ODE_function(z + 0.5*dt*k2)
            k4 = self.geodesic_ODE_function(z + dt*k3)

            z = z + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

            return z

        #step function needed for the jax.lax.scan call
        def step_function(z, _):
            z_next = RK4_step(z, dt)
            return z_next, None

        #use jax.lax.scan to run the integration loop
        z, _ = jax.lax.scan(step_function, z, None, length=num_steps) #later maybe checkpoint for saving memory

        return z


    #same as exp above, but this version returns the whole geodesic trajectory
    def exp_return_trajectory(self, z, t, num_steps: int):

        #perform a single Runge Kutta 4 time step
        def RK4_step(z, dt):
            k1 = self.geodesic_ODE_function(z)
            k2 = self.geodesic_ODE_function(z + 0.5 * dt * k1)
            k3 = self.geodesic_ODE_function(z + 0.5 * dt * k2)
            k4 = self.geodesic_ODE_function(z + dt * k3)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            return z

        #step function needed for the jax.lax.scan call, now returns each intermediate z value
        def step_function(z, _):
            z_next = RK4_step(z, dt)
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


    #forward pass.
    #expect size of y_in to be of shape (dim_dataspace,) and t of shape () or (1,)
    def __call__(self, y_in, t, num_steps : int):

        #go from the dataspace y in R^n to the tangentbundle to z in TM of shape (1,dim_M) or (dim_M,)
        z_in = self.psi(y_in)

        #evaluate the exponential map on z in TM to new z in TM of shape (1,dim_M) or (dim_M,)
        z_out = self.exp(z_in, t, num_steps)

        #immerse z in TM back into the dataspace to y in R^n of shape (1,dim_dataspace) or (dim_dataspace,)
        y_out = self.phi(z_out)

        return y_out


    #Riemann curvature tensor. expect input x in M. The output has shape (m,m,m,m)
    def Riemann_curvature(self, x):

        G = lambda x : self.connection_coeffs(x)

        Gx = G(x)

        #formula R^i_jkl = partial_k G^i_lj - partial_l G^i_kj + G^i_kp G^p_lj - G^i_lp G^p_kj

        #compute partial derivatives of G w.r.t. x, shape (m,m,m,m), partial in the last index
        partialG = jax.jacfwd(G)(x)
        #now the partial is in the first index
        partialG = jnp.transpose(partialG, axes=(3, 0, 1, 2))

        #compute each summand in the Riemann curvature tensor formula

        #partial_k G^i_lj
        term1 = jnp.einsum('kilj->ijkl',partialG, optimize="optimal")

        #partial_l G^i_kj
        term2 =  jnp.einsum('likj->ijkl',partialG, optimize="optimal")

        #G^i_kp G^p_lj
        term3 = jnp.einsum('ikp,plj->ijkl', Gx, Gx, optimize="optimal")

        #G^i_lp G^p_kj
        term4 = jnp.einsum('ilp,pkj->ijkl', Gx, Gx, optimize="optimal")

        #combine terms to get the Riemann curvature tensor
        R = term1 - term2 + term3 - term4  # Shape: (m, m, m, m)

        return R

    #Ricci curvature tensor. expect input x in M. The output has shape (m,m,)
    def Ricci_curvature(self, x):

        R = self.Riemann_curvature(x)

        Ric = jnp.einsum('lilj->ij',R, optimize="optimal")

        return Ric

    #scalar curvature. expect input x in M. The output has shape ()
    def scalar_curvature(self, x):

        #get the Ricci Ric_ij dx^i tensor dx^j
        Ric = self.Ricci_curvature(x)

        #get the inverse metric g^ij partial_i  tensor partial_j
        g_inv = jnp.linalg.inv(self.g(x))

        scal = jnp.einsum('ij,ij->',g_inv,Ric, optimize="optimal")

        return scal

    #sectional curvature if dim M = 2. expect input x in M. The output has shape ()
    def sectional_curvature(self, x):

        assert self.dim_M == 2, f"The sectional curvature can only be calculated in this way if dim M = 2 but got dim M = {self.dim_M}"

        g = self.g(x)

        R = self.Riemann_curvature(x)

        R_1212 = jnp.sum(g[0,:]*R[:,1,0,1])

        return R_1212/jnp.linalg.det(g)
