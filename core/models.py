"""
The class TangentBundle contains the main geometric functionally of NGFs,
which are determined through the autoencoder psi, phi and the metric g.

For the class to be modular each of these is passable at initialization,
so that they can be hard coded functions or neural networks.

All methods and calculations are done mathematically, i.e. not with data arrays
but rather with the exact input format that the methods are mathematically defined for (single points or vectors).
"""

import jax
import jax.numpy as jnp

import equinox as eqx


class TangentBundle(eqx.Module):

    #class representing the tangent bundle TM with options for hardcoded or learnable functions psi, phi, g.

    #concretely TM is expressed in a chart TU. m = dim M, 2m = dim TM.

    #points in TU are z=(x,v) in (U subset R^m) x R^m.

    #g in Gamma(TU tensor TU) is the metric tensor on U.

    psi: callable
    phi: callable
    g: callable

    dim_dataspace : int
    dim_M : int

    #upon changing arguments, also adapt get_high_level_parameters, always !
    def __init__(self, dim_dataspace, dim_M, psi , phi , g):

        self.dim_dataspace = dim_dataspace
        self.dim_M = dim_M

        self.psi = psi
        self.phi = phi
        self.g = g


    #give all the high level parameters as a dictionary. this is used in applications/utils to load a saved instance
    def get_high_level_parameters(self):
        params = {
            'dim_dataspace' : self.dim_dataspace,
            'dim_M': self.dim_M,
            'psi_neural_network_classname' : self.psi.classname,
            'phi_neural_network_classname' : self.phi.classname,
            'g_neural_network_classname' : self.g.classname,
            'psi_arguments': self.psi.arguments,
            'phi_arguments': self.phi.arguments,
            'g_arguments': self.g.arguments,
            }

        return params


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
        term1 = jnp.einsum('ki,aib->kab', inverse_g, partial_g)  # 1st term: partial_b g_ia
        term2 = jnp.einsum('ki,bai->kab', inverse_g, partial_g)  # 2nd term: partial_a g_ib
        term3 = jnp.einsum('ki,iab->kab', inverse_g, partial_g)  # 3rd term: partial_i g_ab

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

        dvbydt = -jnp.einsum('kab, a, b -> k', Gamma, v, v) # -Gamma^k_{ab} v^a v^b

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
        term1 = jnp.einsum('kilj->ijkl',partialG)

        #partial_l G^i_kj
        term2 =  jnp.einsum('likj->ijkl',partialG)

        #G^i_kp G^p_lj
        term3 = jnp.einsum('ikp,plj->ijkl', Gx, Gx)

        #G^i_lp G^p_kj
        term4 = jnp.einsum('ilp,pkj->ijkl', Gx, Gx)

        #combine terms to get the Riemann curvature tensor
        R = term1 - term2 + term3 - term4  # Shape: (m, m, m, m)

        return R

    #Ricci curvature tensor. expect input x in M. The output has shape (m,m,)
    def Ricci_curvature(self, x):

        R = self.Riemann_curvature(x)

        Ric = jnp.einsum('lilj->ij',R)

        return Ric

    #scalar curvature. expect input x in M. The output has shape ()
    def scalar_curvature(self, x):

        #get the Ricci Ric_ij dx^i tensor dx^j
        Ric = self.Ricci_curvature(x)

        #get the inverse metric g^ij partial_i  tensor partial_j
        g_inv = jnp.linalg.inv(self.g(x))

        scal = jnp.einsum('ij,ij->',g_inv,Ric)

        return scal

    #sectional curvature if dim M = 2. expect input x in M. The output has shape ()
    def sectional_curvature(self, x):

        assert self.dim_M == 2, f"The sectional curvature can only be calculated in this way if dim M = 2 but got dim M = {self.dim_M}"

        g = self.g(x)

        R = self.Riemann_curvature(x)

        R_1212 = jnp.sum(g[0,:]*R[:,1,0,1])

        return R_1212/jnp.linalg.det(g)
