"""
Holds examples of phi, psi, g of several tangentbundles TM.

The notation is x in M, z=(x,v) in TM, r in R^n (immersed).

Each (non Euclidean) manifold has a method chartdomain, returning the limits and names of x_1,...,x_m.

For some manifolds there are more example functions,
like curvature and so on. These are hard coded such that
they are guaranteed to be correct and can be used for tests.
"""

import jax
import jax.numpy as jnp

import equinox as eqx

import numpy as np

"""
real Eucleadian space with chart R^n and metric g = id
"""
def psi_id(z):
    r = z
    return r

def phi_id(r):
    z = r
    return z

#expect x of shape (dim,)
def g_id(x):

    dim = x.shape[0]

    g = jnp.eye(dim)

    return g

"""
twosphere S^2 in the chart (0,pi)_curlytheta x (-pi,pi)_curlyphi
with metric

g(x) = ( 1         0         )
       ( 0 sin^2(curlytheta) )
"""
def chartdomain_S2_spherical():
    #return theta min, theta max, phi min, phi max, x^0 name, x^1 name
    return 0.001*jnp.pi, 0.999*jnp.pi, -0.999*jnp.pi, 0.999*jnp.pi, r'\vartheta', r'\varphi'

#the inverse of the embedding on its image.
#this was obtained by first inverting the parametrization and then vy^1 = dpara (vctheta, vcphi)^1, vy^3 = dpara (vctheta, vcphi)^3
#y = (y^1,y^2,y^3,vy^1,vy^2,vy^3)
def psi_S2_spherical(r):

    x,y,z,vx,vy,vz = r[0],r[1],r[2],r[3],r[4],r[5]

    theta = jnp.arccos(z)
    phi = jnp.arctan2(y,x)

    v_theta = -vz/jnp.sin(theta)
    v_phi = -1/jnp.sin(theta)*(vz/(jnp.tan(theta)*jnp.tan(phi)) + vx/jnp.sin(phi))

    z = jnp.array([theta,phi,v_theta, v_phi])

    return z

def parametrization_S2_spherical(x):

    theta = x[0]
    phi = x[1]

    return jnp.array([jnp.sin(theta)*jnp.cos(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(theta)])


def phi_S2_spherical(z):

    x = z[0:2]
    v = z[2:4]

    #take the jacobian of the parametrization
    dparametrization_S2_spherical = jax.jacfwd(parametrization_S2_spherical)

    #the first three components hold the position and the last three the velocity
    r = jnp.zeros(6)

    r = r.at[0:3].set(parametrization_S2_spherical(x))
    r = r.at[3:6].set(dparametrization_S2_spherical(x) @ v)

    return r

def g_S2_spherical(x):
    return jnp.array([[1,0],[0,jnp.sin(x[0])**2]])

def scalarproduct_S2_spherical(x,u,v):
    return u @ g_S2_spherical(x) @ v

"""
twosphere S^2, in the chart R^2 stereographic projection
with metric

g(x) = ( 4/h^2     0  )
       ( 0      4/h^2 )

where h(x) = 1 + x0^2 + x1^2
"""
def chartdomain_S2_stereographic():
    #since R^2 is unbounded, just return some large bounds, x^0 name, x^1 name
    return -1.2, 1.2, -1.2, 1.2, r'x^0', r'x^1'

#I have calculated derivates manually. These exclude the north pole
def psi_S2_stereographic(r):

        x,y,z,vx,vy,vz = r[0],r[1],r[2],r[3],r[4],r[5]

        return jnp.array([x/(1-z),y/(1-z),vx/(1-z) + x*vz/((1-z)**2), vy/(1-z) + y*vz/((1-z)**2)])

#these exclude the south pole instead (swapped signs in front of z and vz)
def psi_S2_inverted_stereographic(r):

        x, y, z, vx, vy, vz = r[0], r[1], r[2], r[3], r[4], r[5]

        return jnp.array([
            x / (1 + z),
            y / (1 + z),
            vx / (1 + z) - x * vz / ((1 + z) ** 2),
            vy / (1 + z) - y * vz / ((1 + z) ** 2)
        ])


#I calculated the derivates manually. These exclude the north pole
def phi_S2_stereographic(z):

        x = z[0:2]
        v = z[2:4]
        h = 1 + x[0] ** 2 + x[1] ** 2

        r0 = 2 * x[0] / h
        r1 = 2 * x[1] / h
        r2 = (h - 2) / h
        r3 = (2 - 2 * x[0] ** 2 + 2 * x[1] ** 2) / (h ** 2) * v[0] - 4 * x[0] * x[1] / (h ** 2) * v[1]
        r4 = -4 * x[0] * x[1] / (h ** 2) * v[0] + (2 + 2 * x[0] ** 2 - 2 * x[1] ** 2) / (h ** 2) * v[1]
        r5 = 4 * x[0] / (h ** 2) * v[0] + 4 * x[1] / (h ** 2) * v[1]

        return jnp.array([r0, r1, r2, r3, r4, r5])


#these exclude the south pole instead (just swapped signs in r2 and r5)
def phi_S2_inverted_stereographic(z):

        x = z[0:2]
        v = z[2:4]
        h = 1 + x[0] ** 2 + x[1] ** 2

        r0 = 2 * x[0] / h
        r1 = 2 * x[1] / h
        r2 = -(h - 2) / h
        r3 = (2 - 2 * x[0] ** 2 + 2 * x[1] ** 2) / (h ** 2) * v[0] - 4 * x[0] * x[1] / (h ** 2) * v[1]
        r4 = -4 * x[0] * x[1] / (h ** 2) * v[0] + (2 + 2 * x[0] ** 2 - 2 * x[1] ** 2) / (h ** 2) * v[1]
        r5 = -4 * x[0] / (h ** 2) * v[0] - 4 * x[1] / (h ** 2) * v[1]

        return jnp.array([r0, r1, r2, r3, r4, r5])


#the metric is the same in the default and inverted case
def g_S2_stereographic(x):

        h = 1 + x[0] ** 2 + x[1] ** 2

        return jnp.array([
            [4 / (h ** 2), 0.0],
            [0.0, 4 / (h ** 2)]
        ])


"""
two sphere S^2 in the chart (-2,2)_xi x (-2,2)_eta normal coordinates.

The coordinates are linked to the usual theta, phi like

xi = theta cos(phi),
eta = theta sin(phi).

The metric is complicated.
"""
def chartdomain_S2_normal():
    #these are not definitive bounds but they work. zero is a singularity.
    return -2, 2, -2, 2, r'$\xi$', r'$\eta$'

def parametrization_S2_normal(x):

    xi = x[0]
    eta = x[1]

    y1 = (xi*jnp.sin(jnp.sqrt(eta**2 + xi**2)))/jnp.sqrt(eta**2 + xi**2)
    y2 = (eta*jnp.sin(jnp.sqrt(eta**2 + xi**2)))/jnp.sqrt(eta**2 + xi**2)
    y3 = jnp.cos(jnp.sqrt(eta**2 + xi**2))

    return jnp.array([y1,y2,y3])

#inverse function from the embedding to the chart
def psi_S2_normal(r):

    x,y,z = r[0],r[1],r[2]

    #auxiliary function
    h = jnp.arccos(z)

    #solved for xi and eta
    xi = h/jnp.sin(h)*x
    eta = h/jnp.sin(h)*y

    #jacobian of the parametrization
    dparametrization = jax.jacfwd(parametrization_S2_normal)

    #now we solve dpara @ (vxi,veta) = (vx,vy,vz) by choosing randomly the last two rows
    A = dparametrization(jnp.array([xi,eta]))[1:3,:]

    #solve for vxi, veta using only vy,vz
    v = jnp.linalg.inv(A)@r[4:6]

    vxi,veta = v[0],v[1]

    return jnp.array([xi,eta,vxi,veta])

#embedding into R^6
def phi_S2_normal(z):

    x = z[0:2]
    v = z[2:4]

    #jacobian of the parametrization
    dparametrization = jax.jacfwd(parametrization_S2_normal)

    #the first three components hold the position and the last three the velocity
    r = jnp.zeros(6)

    r = r.at[0:3].set(parametrization_S2_normal(x))
    r = r.at[3:6].set(dparametrization(x) @ v)

    return r

def g_S2_normal(x):

    xi = x[0]
    eta = x[1]

    g_xixi = xi**2/(eta**2 + xi**2) + (eta**2*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2
    g_xieta = (eta*xi)/(eta**2 + xi**2) - (eta*xi*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2
    g_etaxi = (eta*xi)/(eta**2 + xi**2) - (eta*xi*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2
    g_etaeta = eta**2/(eta**2 + xi**2) + (xi**2*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2

    return jnp.array([[g_xixi,g_xieta],[g_etaxi,g_etaeta]])

#the inverse (dual) metric of g at x = (xi, eta)
def g_inverse_S2_normal(x):
    return jnp.inv(g_S2_normal(x))

def scalarproduct_S2_normal(x,u,v):
    return u @ g_S2_normal(x) @ v

def det_g_S2_normal(x):
    return jnp.linalg.det(g_S2_normal(x))

#the volume form is a top dimensional differential form vol = sqrt(det g_x)dx^0 wedge dx^1 and is evaluated at tangentvectors u,v.
#recall that by definiton                          ( dx(u) dy(u) )
#                           dx wedge dy (u,v) = det( dx(v) dy(v) )
def volumeform_S2_normal(x,u,v):
    return jnp.sqrt(jnp.linalg.det(g_S2_normal(x)))*(u[0]*v[1]-u[1]*v[0])

#the Christoffel symbols
def connection_coeffs_S2_normal(x):

    xi = x[0]
    eta  = x[1]

    #holds Gamma^k_ab in [k,a,b]
    Gamma = jnp.zeros((2,2,2))

    Gamma_xi_xixi = (eta**2*xi*(-jnp.sqrt(eta**2 + xi**2) + 2*(eta**2 + xi**2)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) - jnp.sin(2*jnp.sqrt(eta**2 + xi**2))/2))/(eta**2 + xi**2)**(5/2)
    Gamma_xi_xieta = (-(eta**3*jnp.sqrt(eta**2 + xi**2)) + eta*(eta**4 - xi**4)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) + (eta*xi**2*jnp.sin(2*jnp.sqrt(eta**2 + xi**2)))/2)/(eta**2 + xi**2)**(5/2)
    Gamma_xi_etaxi = (-(eta**3*jnp.sqrt(eta**2 + xi**2)) + eta*(eta**4 - xi**4)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) + (eta*xi**2*jnp.sin(2*jnp.sqrt(eta**2 + xi**2)))/2)/(eta**2 + xi**2)**(5/2)
    Gamma_xi_etaeta = (xi*(jnp.sqrt(eta**2 + xi**2)*(2*eta**2 + xi**2) - 2*eta**2*(eta**2 + xi**2)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) - (xi**2*jnp.sin(2*jnp.sqrt(eta**2 + xi**2)))/2))/(eta**2 + xi**2)**(5/2)

    Gamma_eta_xixi = (eta*(jnp.sqrt(eta**2 + xi**2)*(eta**2 + 2*xi**2) - 2*xi**2*(eta**2 + xi**2)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) - (eta**2*jnp.sin(2*jnp.sqrt(eta**2 + xi**2)))/2))/(eta**2 + xi**2)**(5/2)
    Gamma_eta_xieta = (-(xi**3*jnp.sqrt(eta**2 + xi**2)) + (-(eta**4*xi) + xi**5)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) + (eta**2*xi*jnp.sin(2*jnp.sqrt(eta**2 + xi**2)))/2)/(eta**2 + xi**2)**(5/2)
    Gamma_eta_etaxi = (-(xi**3*jnp.sqrt(eta**2 + xi**2)) + (-(eta**4*xi) + xi**5)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) + (eta**2*xi*jnp.sin(2*jnp.sqrt(eta**2 + xi**2)))/2)/(eta**2 + xi**2)**(5/2)
    Gamma_eta_etaeta = (eta*xi**2*(-jnp.sqrt(eta**2 + xi**2) + 2*(eta**2 + xi**2)*1/jnp.tan(jnp.sqrt(eta**2 + xi**2)) - jnp.sin(2*jnp.sqrt(eta**2 + xi**2))/2))/(eta**2 + xi**2)**(5/2)

    #Gamma^theta
    Gamma = Gamma.at[0,:,:].set(jnp.array([[Gamma_xi_xixi,Gamma_xi_xieta],[Gamma_xi_etaxi,Gamma_xi_etaeta]]))

    #Gamma^phi
    Gamma = Gamma.at[1,:,:].set(jnp.array([[Gamma_eta_xixi,Gamma_eta_xieta],[Gamma_eta_etaxi,Gamma_eta_etaeta]]))

    return Gamma


def geodesic_ODE_function_S2_normal(z):

    xi = z[0]
    eta = z[1]

    x = jnp.array([xi,eta])

    v_xi = z[2]
    v_eta = z[3]

    #geodesic ODE
    dxibydt = v_xi
    detabydt = v_eta

    dv_xibydt = - v_xi**2*connection_coeffs_S2_normal(x)[0,0,0] - 2*v_xi*v_eta*connection_coeffs_S2_normal(x)[0,0,1] - v_eta**2*connection_coeffs_S2_normal(x)[0,1,1]
    dv_etabydt = - v_xi**2*connection_coeffs_S2_normal(x)[1,0,0] - 2*v_xi*v_eta*connection_coeffs_S2_normal(x)[1,0,1] - v_eta**2*connection_coeffs_S2_normal(x)[1,1,1]

    return jnp.array([dxibydt, detabydt, dv_xibydt, dv_etabydt])

def exp_S2_normal(z, t, num_steps : int):

    #perform a single Runge Kutta 4 time step
    def RK4_step(z, dt):
        k1 = geodesic_ODE_function_S2_normal(z)
        k2 = geodesic_ODE_function_S2_normal(z + 0.5*dt*k1)
        k3 = geodesic_ODE_function_S2_normal(z + 0.5*dt*k2)
        k4 = geodesic_ODE_function_S2_normal(z + dt*k3)

        z = z + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

        return z

    dt = t/num_steps

    for i in range(num_steps):

        z = RK4_step(z, dt)

    return z


def exp_return_trajectory_S2_normal(z, t, num_steps : int):

    #perform a single Runge Kutta 4 time step
    @jax.jit
    def RK4_step(z, dt):
        k1 = geodesic_ODE_function_S2_normal(z)
        k2 = geodesic_ODE_function_S2_normal(z + 0.5*dt*k1)
        k3 = geodesic_ODE_function_S2_normal(z + 0.5*dt*k2)
        k4 = geodesic_ODE_function_S2_normal(z + dt*k3)

        z = z + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

        return z

    dt = t/num_steps

    z_geo = jnp.zeros((num_steps+1,4))
    z_geo = z_geo.at[0,:].set(z)

    for i in range(num_steps):

        z = RK4_step(z, dt)

        z_geo = z_geo.at[i+1,:].set(z)

    return z_geo

def get_geodesic_S2_normal(r, t, num_steps : int):

    z = psi_S2_normal(r)

    z_geo = exp_return_trajectory_S2_normal(z, t, num_steps)

    r_geo = jax.vmap(phi_S2_normal, in_axes = 0)(z_geo)

    return r_geo

def Riemann_curvature_S2_normal(x):

    xi = x[0]
    eta = x[1]

    #copied from my Mathematica script
    R_xixi_xixi = 0
    R_xixi_xieta = (eta*xi*(eta**2 + xi**2 - jnp.sin(jnp.sqrt(eta**2 + xi**2))**2))/(eta**2 + xi**2)**2
    R_xixi_etaxi = - R_xixi_xieta
    R_xixi_etaeta = 0

    R_xieta_xixi = 0
    R_xieta_xieta = (eta**4 + eta**2*xi**2 + xi**2*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2
    R_xieta_etaxi = - R_xieta_xieta
    R_xieta_etaeta = 0

    R_etaxi_xixi = 0
    R_etaxi_xieta = -((eta**2*xi**2 + xi**4 + eta**2*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2)
    R_etaxi_etaxi = - R_etaxi_xieta
    R_etaxi_etaeta = 0

    R_etaeta_xixi = 0
    R_etaeta_xieta = -((eta*xi*(eta**2 + xi**2 - jnp.sin(jnp.sqrt(eta**2 + xi**2))**2))/(eta**2 + xi**2)**2)
    R_etaeta_etaxi = - R_etaeta_xieta
    R_etaeta_etaeta = 0

    R = jnp.zeros((2,2,2,2))

    R = R.at[0,0,0,0].set(R_xixi_xixi)
    R = R.at[0,0,0,1].set(R_xixi_xieta)
    R = R.at[0,0,1,0].set(R_xixi_etaxi)
    R = R.at[0,0,1,1].set(R_xixi_etaeta)

    R = R.at[0,1,0,0].set(R_xieta_xixi)
    R = R.at[0,1,0,1].set(R_xieta_xieta)
    R = R.at[0,1,1,0].set(R_xieta_etaxi)
    R = R.at[0,1,1,1].set(R_xieta_etaeta)

    R = R.at[1,0,0,0].set(R_etaxi_xixi)
    R = R.at[1,0,0,1].set(R_etaxi_xieta)
    R = R.at[1,0,1,0].set(R_etaxi_etaxi)
    R = R.at[1,0,1,1].set(R_etaxi_etaeta)

    R = R.at[1,1,0,0].set(R_etaeta_xixi)
    R = R.at[1,1,0,1].set(R_etaeta_xieta)
    R = R.at[1,1,1,0].set(R_etaeta_etaxi)
    R = R.at[1,1,1,1].set(R_etaeta_etaeta)

    return R

def Ricci_curvature_S2_normal(x):

    xi = x[0]
    eta = x[1]

    #copied from Mathematica
    Ric_xixi = (eta**2*xi**2 + xi**4 + eta**2*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2
    Ric_xieta = (eta*xi*(eta**2 + xi**2 - jnp.sin(jnp.sqrt(eta**2 + xi**2))**2))/(eta**2 + xi**2)**2
    Ric_etaxi = Ric_xieta
    Ric_etaeta = (eta**4 + eta**2*xi**2 + xi**2*jnp.sin(jnp.sqrt(eta**2 + xi**2))**2)/(eta**2 + xi**2)**2

    Ric = jnp.zeros((2,2))

    Ric = Ric.at[0,0].set(Ric_xixi)
    Ric = Ric.at[0,1].set(Ric_xieta)
    Ric = Ric.at[1,0].set(Ric_etaxi)
    Ric = Ric.at[1,1].set(Ric_etaeta)

    return Ric

def scalar_curvature_S2_normal(x):

    return 2.0

def sectional_curvature_S2_normal(x):

    return 1.0


class two_body_Jacobi_metric(eqx.Module):
    #Jacobi_metric. Parametrically depends on total_energy

    #hardcoded for 2-body problem with masses equal to one.

    #needs to be pulled back to the chart
    #because it is apriori only defined in the dataspace of dimension 4 (tangentbundle dimension 8)
    #hence we need access to the parametrization function phi

    phi : callable

    total_energy : float

    arguments : dict
    classname : str

    #dictionary "arguments" has to hold dim_dataspace (int)
    def __init__(self, phi, arguments, key = jax.random.PRNGKey(0)):

        #verify that essentialkeys are provided
        required_keys = ['total_energy']
        for dict_key in required_keys:
            if dict_key not in arguments:
                raise ValueError(f"Missing required key '{dict_key}' in arguments")


        #assign member variables
        self.phi = phi

        self.total_energy = arguments['total_energy']

        self.arguments = arguments
        self.classname = "two_body_Jacobi_metric"


    #in case we modify the chart (phi) after having defined an instance of this class
    #we need to be able to update phi
    def update_phi(self, phi):

        self.phi = phi

    #pass yx in R^4 from y = (yx,yv) in R^8, yx holding the locations of the two bodies A and B
    def potential_energy(self, yx):
        A = yx[0:2]
        B = yx[2:4]

        return -1/jnp.linalg.norm(A-B)

    #pass yx in R^4 from y = (yx,yv) in R^8, rx holding the locations of the two bodies A and B
    #masses are one, so the kinetic scalarproduct is the standard scalar product. Hence:
    #g = 2(h - U)id
    def analytical_metric(self, yx):

        factor = 2*(self.total_energy - self.potential_energy(yx))

        return factor*jnp.eye(4)

    #expect x from the chart in R^2
    def __call__(self, x):

        #artifical zero gradients, f : R^2 (chart) -----> R^4 (locations in dataspace)
        f = lambda x : self.phi(jnp.concatenate([x, jnp.zeros((2,))]))[0:4]

        df = jax.jacfwd(f)

        #pullback formula g = df^T(x) * analytical_metric(f(x)) * df(x)
        yx = f(x)

        #analytical_metric at yx = f(x)
        g_yx = self.analytical_metric(yx)

        #pullback metric at x
        g_x = jnp.transpose(df(x)) @ g_yx @ df(x)

        return g_x

"""
two torus T^2 in the chart (0,2pi)_x0 x (0,2pi)_x1
where x0 is the little circle (radius b) and x1 is the big circle (radius a)
with metric

g(x) = ( b^2         0        )
       ( 0    (a + b cos x^0)^2 )
"""
b = 0.5
a = 1.5

def chartdomain_T2():
    #return x0 min, x0 max, x1 min, x1 max, x^0 name, x^1 name
    return 0.001*jnp.pi, 1.999*jnp.pi, 0.001*jnp.pi, 1.999*jnp.pi, r'x^0', r'x^1'

#the inverse of the embedding on its image.
def psi_T2(r):

    x,y,z,vx,vy,vz = r[0],r[1],r[2],r[3],r[4],r[5]

    print("Warning: psi_T2 is not properly implemented, yields 0")

    z = jnp.array([0, 0, 0, 0])

    return z

def parametrization_T2(x):

    return jnp.array([(a + b*jnp.cos(x[0]))*jnp.cos(x[1]), (a + b*jnp.cos(x[0]))*jnp.sin(x[1]), b*jnp.sin(x[0])])


def phi_T2(z):

    x = z[0:2]
    v = z[2:4]

    #take the jacobian of the parametrization
    dparametrization_T2 = jax.jacfwd(parametrization_T2)

    #the first three components hold the position and the last three the velocity
    r = jnp.zeros(6)

    r = r.at[0:3].set(parametrization_T2(x))
    r = r.at[3:6].set(dparametrization_T2(x) @ v)

    return r

def g_T2(x):
    return jnp.array([[b**2,0],[0,(a + b*jnp.cos(x[0]))**2]])


"""
Gaussian statistical manifold in the chart (-inf,inf)_mu x [0,inf)_sigma
where mu is the mean and sigma the standard deviation
with metric

g(x) = ( 1/sigma^2      0     )
       ( 0          2/sigma^2 )
"""

def chartdomain_Gaussians1():
    #return x0 min, x0 max, x1 min, x1 max, x^0 name, x^1 name
    return -jnp.inf, jnp.inf, 0, jnp.inf, r'\mu', r'\sigma'

#the inverse of the embedding on its image.
def psi_Gaussians1(r):

    x,y,z,vx,vy,vz = r[0],r[1],r[2],r[3],r[4],r[5]

    print("Warning: psi_Gaussians1 is not properly implemented, yields 0")

    z = jnp.array([0, 0, 0, 0])

    return z

def parametrization_Gaussians1(x):

    mu = x[0]
    sigma = x[1]

    return jnp.array([mu,sigma])


def phi_Gaussians1(z):

    x = z[0:2]
    v = z[2:4]

    #take the jacobian of the parametrization
    dparametrization_Gaussians1 = jax.jacfwd(parametrization_Gaussians1)

    #the first three components hold the position and the last three the velocity
    r = jnp.zeros(6)

    r = r.at[0:3].set(parametrization_Gaussians1(x))
    r = r.at[3:6].set(dparametrization_Gaussians1(x) @ v)

    return r

def g_Gaussians1(x):
    return jnp.array([[1/(x[1]**2),0],[0,2/(x[1]**2)]])
