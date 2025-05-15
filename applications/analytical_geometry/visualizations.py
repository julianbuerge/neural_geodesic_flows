"""
Visualization methods for analytical geometry data
"""

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

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
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
})

def chart_geodesic_visualization(tangentbundle, geodesics, chartdomain, grid_res = (30,30), name = 'manifold'):

    x0_min, x0_max, x1_min, x1_max, x0_name, x1_name = chartdomain()

    ######## plotting #######
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(111)

    #plot the chart bounding box


    #plot the geodesics
    if geodesics.ndim == 2:
        ax1.plot(geodesics[:, 0], geodesics[:, 1], color='black', label='geodesic')
    else:
        ax1.plot(geodesics[:,:, 0], geodesics[:,:, 1], color='black', label='geodesics')

    #plot the initial point
    if geodesics.ndim == 2:
        ax1.scatter(geodesics[0,0], geodesics[0,1], color='red', marker = 'o', label='initial')
        ax1.quiver(geodesics[0,0], geodesics[0,1],
                   geodesics[0,2], geodesics[0,3],
                   color='red')
    else:
        ax1.scatter(geodesics[:,0,0], geodesics[:,0,1], color='red', marker = 'o', label='initials')
        ax1.quiver(geodesics[:,0,0], geodesics[:,0,1],
                   geodesics[:,0,2], geodesics[:,0,3],
                   color='red')

    ax1.set_xlabel(r'$'+x0_name+'$')
    ax1.set_ylabel(r'$'+x1_name+'$')
    ax1.set_title(f'Geodesics in a chart of a {name}')
    ax1.legend()
    ax1.axis('equal')

    plt.tight_layout()
    plt.show()

def parametrized_geodesic_visualization(tangentbundle, geodesics, chartdomain, grid_res = (30,30), name = 'manifold'):

    #embed the manifold into 3d
    x0_min, x0_max, x1_min, x1_max, x0_name, x1_name = chartdomain()

    #create grid
    x0_vals = jnp.linspace(x0_min, x0_max, grid_res[0])
    x1_vals = jnp.linspace(x1_min, x1_max, grid_res[1])
    grid = jnp.array(jnp.meshgrid(x0_vals, x1_vals))

    #flatten the grid to shape (res0*res1, 2)
    grid_flat = grid.reshape(2, -1).T

    #add articifical tangents
    grid_tangents = jnp.zeros_like(grid_flat)

    #map the grid points into the data space shape (res0*res1, 3)
    mapped_grid = jax.vmap(tangentbundle.phi, in_axes=0)(jnp.concatenate([grid_flat, grid_tangents], axis = 1))

    #reshape mapped grid back to match grid structure (res0, res1, 3)
    mapped_grid = mapped_grid.reshape(grid_res[0], grid_res[1], -1)

    #extract x, y, z for plotting
    x_grid, y_grid, z_grid = mapped_grid[..., 0], mapped_grid[..., 1], mapped_grid[..., 2]


    ######## plotting #######
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot_wireframe(x_grid, y_grid, z_grid, color="gray", alpha=0.4, label = f'{name}')

    #plot the geodesics
    if geodesics.ndim == 2:
        ax1.plot(geodesics[:, 0], geodesics[:, 1], geodesics[:, 2], color='black', label='geodesic')
    else:
        ax1.plot(geodesics[:,:, 0], geodesics[:,:, 1], geodesics[:,:, 2], color='black', label='geodesics')

    #plot the initial point
    if geodesics.ndim == 2:
        ax1.scatter(geodesics[0,0], geodesics[0,1], geodesics[0,2], color='red', marker = 'o', label='initial')
        ax1.quiver(geodesics[0,0], geodesics[0,1], geodesics[0,2],
                   geodesics[0,3], geodesics[0,4], geodesics[0,5],
                   color='red', length = 0.25)
    else:
        ax1.scatter(geodesics[:,0,0], geodesics[:,0,1], geodesics[:,0,2], color='red', marker = 'o', label='initials')
        ax1.quiver(geodesics[:,0,0], geodesics[:,0,1], geodesics[:,0,2],
                   geodesics[:,0,3], geodesics[:,0,4], geodesics[:,0,5],
                   color='red', length = 0.25)

    ax1.set_xlabel(r'$y^1$')
    ax1.set_ylabel(r'$y^2$')
    ax1.set_zlabel(r'$y^3$')
    ax1.set_title(f'Geodesics on a {name}')
    ax1.legend()
    ax1.axis('equal')

    plt.tight_layout()
    plt.show()
