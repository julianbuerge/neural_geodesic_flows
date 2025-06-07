"""
Inference methods for multi-chart NGFs. These are currently a bit ad-hoc to test what we've implemented so far.
"""

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
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
    "axes3d.mouserotationstyle": "azel",
})

#return x,y,z for plotting a 2d surface in 3d. Pass a parametrization function
def parametrized_surface(parametrization, chartdomain, grid_res = (30,30)):

    #embed the manifold into 3d
    x0_min, x0_max, x1_min, x1_max, x0_name, x1_name = chartdomain()

    #create grid
    x0_vals = jnp.linspace(x0_min, x0_max, grid_res[0])
    x1_vals = jnp.linspace(x1_min, x1_max, grid_res[1])
    grid = jnp.array(jnp.meshgrid(x0_vals, x1_vals))

    #flatten the grid to shape (res0*res1, 2)
    grid_flat = grid.reshape(2, -1).T

    #map the grid points into the data space shape (res0*res1, 3)
    mapped_grid = jax.vmap(parametrization, in_axes=0)(grid_flat)

    #reshape mapped grid back to match grid structure (res0, res1, 3)
    mapped_grid = mapped_grid.reshape(grid_res[0], grid_res[1], -1)

    #extract x, y, z for plotting
    x_grid, y_grid, z_grid = mapped_grid[..., 0], mapped_grid[..., 1], mapped_grid[..., 2]

    surface = (x_grid, y_grid, z_grid)

    return surface

def full_dynamics_visualization(tangent_bundle,
                                 initial_state,
                                 t,
                                 steps,
                                 surface):  # surface = (x_grid, y_grid, z_grid)

    #integration, in the chart, yielding a tuple (chart_ids, z_values)
    chart_geodesic = tangent_bundle.exp_return_trajectory(initial_state, t, steps)

    #embed the integrated curve into data space, this is now an array of shape (steps+1, 6) (x,y,z,vx,vy,vz)
    geodesic = jax.vmap(tangent_bundle.phi, in_axes = 0)(chart_geodesic)

    #find the surface x,y,z for plotting
    x_grid, y_grid, z_grid = surface


    #prepare chart arrays
    chart_ids, zs = chart_geodesic

    #prepare colors
    amount_of_charts = tangent_bundle.amount_of_charts
    cmap = cm.get_cmap("winter", amount_of_charts)
    chart_colors = [cmap(i) for i in range(amount_of_charts)]


    #plot setup with the surface
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.5, hspace=0.6)
    #gs = gridspec.GridSpec(3, 4, figure=fig)  # 3 rows × 4 columns

    # Data space plot spans the top two columns
    ax_main = fig.add_subplot(gs[0:2, 1:3], projection='3d')
    #ax_main = fig.add_subplot(2, 3, 2, projection='3d')  # center top

    x_grid, y_grid, z_grid = surface
    ax_main.plot_wireframe(x_grid, y_grid, z_grid, color="gray", alpha=0.3)

    #plot geodesic in the data space (colors as they will be in the charts)
    for i in range(len(geodesic) - 1):
        cid = int(chart_ids[i])
        seg = geodesic[i:i + 2]
        ax_main.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=chart_colors[cid], linewidth=2)

    #mark initial point and initial tangent vector
    ax_main.scatter(*geodesic[0, :3], color='black', marker='o', label='initial')
    ax_main.quiver(*geodesic[0, :3], *geodesic[0, 3:], color='black', length=0.25)

    ax_main.set_xlabel(r'$y^1$')
    ax_main.set_ylabel(r'$y^2$')
    ax_main.set_zlabel(r'$y^3$')
    ax_main.set_title("Dynamics in data space")
    ax_main.axis('equal')
    ax_main.legend()

    #plot all charts, each with a different color
    rows = (amount_of_charts + 2) // 3
    plot_idx = 4  # subplot index for chart visualizations

    for i in range(amount_of_charts):

        mask = chart_ids == i

        z_i = zs[mask]

        ax = fig.add_subplot(2, 3, plot_idx)

        ax.plot(z_i[:, 0], z_i[:, 1], '.', color=chart_colors[i])

        ax.set_title(f'chart {i}')
        ax.set_xlabel(r'$x^1$')
        ax.set_ylabel(r'$x^2$')
        ax.axis('equal')
        plot_idx += 1

    #plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()
