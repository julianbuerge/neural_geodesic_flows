"""
Here we create a synthetic geodesics dataset on the full two sphere.
This is analogous to the toy problem from the master thesis, just now on the entire sphere.
"""

import jax
import jax.numpy as jnp

import numpy as np

import sys

np.set_printoptions(threshold=sys.maxsize)

#set a backend for interactive plotting
import matplotlib
matplotlib.use('QtAgg')

#customize the figure default style and format
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "savefig.dpi": 100,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
})

import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

from experimental.atlas import (
    Chart
)

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_S2_spherical,
    psi_S2_spherical,
    phi_S2_spherical,
    g_S2_spherical
)

########## define the sphere bundle in standard theta, varphi coordinates ##########
spherechart = Chart(psi = psi_S2_spherical, phi = phi_S2_spherical, g= g_S2_spherical)



########## generate random initial points on the 2 sphere ##########

#generate a random batch of points, first in  [0,1]^4
size = 40000

z_initial = jax.random.uniform(jax.random.PRNGKey(30),
                               minval = jnp.array([0.001, 0, -1, -1]),
                               maxval = jnp.array([0.999, 1, 1, 1]),
                               shape=(size, 4))

#transform to be uniformly distributed on the entire sphere, formula theta = arccos(1-2z_uniform), phi = 2pi*z_uniform
z_initial = z_initial.at[:,0].set(jnp.arccos(1- 2*z_initial[:,0]))
z_initial = z_initial.at[:,1].set(2*jnp.pi*z_initial[:,1])

#transform the tangents to be uniformly distributed in [-1,1]^2
#z_initial = z_initial.at[:,2].set(-1+ 2*z_initial[:,2])
#z_initial = z_initial.at[:,3].set(-1+ 2*z_initial[:,3])

########## generate trajectories ##########
geo = jax.vmap(spherechart.exp_return_trajectory, in_axes = (0,None,None))

z_trajectories = geo(z_initial,1.0,99) #shape (size, 100, 4)

encode_geo = jax.vmap(spherechart.phi, in_axes = 0)
encode_geos = jax.vmap(encode_geo, in_axes = 0)

y_trajectories = encode_geos(z_trajectories) #shape (size, 100, 6)

time = jnp.linspace(0,1.0,100)
times = jnp.tile(time, (size,1))

is_finite = jnp.all(jnp.isfinite(y_trajectories), axis=(1, 2))  # shape: (size,)

is_reasonable = jnp.all(jnp.abs(y_trajectories) < 2, axis=(1, 2))
mask = jnp.logical_and(is_finite, is_reasonable)

#use the mask to filter all related arrays
y_trajectories = y_trajectories[mask]
times = times[mask]

print(y_trajectories.shape)

print(jnp.min(y_trajectories, axis = (0,1)))
print(jnp.max(y_trajectories, axis = (0,1)))


########## save the trajectories and times as .npz file ##########
np.savez("data/datasets/sphere_trajectories_train.npz", trajectories=y_trajectories[0:16384,:,:], times=times[0:16384,:])
np.savez("data/datasets/sphere_trajectories_test.npz", trajectories=y_trajectories[16384:32768,:,:], times=times[16384:32768,:])

#print("Saved it")


########## plot ##########
display_amount = 128

y_initial = y_trajectories[0:display_amount,0,:]
y_final = y_trajectories[0:display_amount,-1,:]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(y_final[:, 0], y_final[:, 1], y_final[:, 2], color='red', marker = 'o', s = 20, label=r'$y_i$')
ax.quiver(y_final[:, 0], y_final[:, 1], y_final[:, 2],
           y_final[:, 3], y_final[:, 4], y_final[:, 5],
           color='red', length = 0.25)

ax.scatter(y_initial[:, 0], y_initial[:, 1], y_initial[:, 2], color='orange', marker = 's', s = 20, label=r'$\bar{y}_i$')
ax.quiver(y_initial[:, 0], y_initial[:, 1], y_initial[:, 2],
           y_initial[:, 3], y_initial[:, 4], y_initial[:, 5],
           color='orange', length = 0.25)
#for i in range(size):
#    ax.plot(y_trajectories[i, :, 0], y_trajectories[i, :, 1], y_trajectories[i, :, 2], color='black')

ax.legend()
ax.set_title('Examples from the training dataset')
ax.axis('equal')
plt.tight_layout()
plt.show()
