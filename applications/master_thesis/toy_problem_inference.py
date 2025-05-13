"""
Replication of the case study "Toy problem on S^2_+" inference
from the master thesis Neural geodesic flows published at
https://doi.org/10.3929/ethz-b-000733724
"""

import jax
import jax.numpy as jnp

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

import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull

#get the relevant neural network classes to initialize psi,phi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    NN_diffeomorphism,
    NN_Jacobian_split_diffeomorphism,
    NN_metric,
)

from core.inference import (
    apply_model_function,
    trajectory_model_analyis
)

#get some loading methods to load models and test datasets
from applications.utils import (
    load_dataset,
    load_model,
)

#choose what models inference to show
show_model_A = False    #input target model, Jacobian split
show_model_B = False    #input target model, no split
show_model_C = True     #best model, trajectory and split
show_model_D = False    #trajectory model, no split

#error analysis shown for the toy problem
def toy_problem_error_analysis(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)

    #load the data, respecting the correct mode
    data, _ = load_dataset(name = "half-sphere_trajectories_test",
                 size=1024,
                 random_selection=True,
                 key=jax.random.PRNGKey(20))

    trajectories, times = data

    trajectory_model_analyis(model, trajectories, times)


#auxilary function for the method below
def show_trajectories(model,
                      data_trajectories, #given trajectories in the data space
                      data_predictions, #predictions in the data space
                      time):

    #collect all points together to visualize the chart domain
    data_points = data_trajectories.reshape(-1, data_trajectories.shape[-1]) #from (many, timesteps, ...) to (many*timesteps, ...)

    #map the data points to the chart, shape (many, 4)
    chart_data_points = jax.vmap(model.psi, in_axes = 0)(data_points)

    #find the chart domain of only U^P shape (many, 2) as a convex hull
    chart_hull = ConvexHull(chart_data_points[:,0:2])

    #create a grid in the chart of shape (2,res0,res1)
    grid_res = (30,30)
    x0_vals = jnp.linspace(chart_data_points[:, 0].min(), chart_data_points[:, 0].max(), grid_res[0])
    x1_vals = jnp.linspace(chart_data_points[:, 1].min(), chart_data_points[:, 1].max(), grid_res[1])
    grid = jnp.array(jnp.meshgrid(x0_vals, x1_vals))
    grid_x0, grid_x1 = jnp.meshgrid(x0_vals, x1_vals)

    #flatten the grid to shape (res0*res1, 2)
    grid_flat = grid.reshape(2, -1).T

    #add articifical tangents
    grid_tangents = jnp.zeros_like(grid_flat)

    #map the grid points into the data space shape (res0*res1, 3)
    mapped_grid = jax.vmap(model.phi, in_axes=0)(jnp.concatenate([grid_flat, grid_tangents], axis = 1))

    #reshape mapped grid back to match grid structure (res0, res1, 3)
    mapped_grid = mapped_grid.reshape(grid_res[0], grid_res[1], -1)

    #extract x, y, z for plotting
    x_grid, y_grid, z_grid = mapped_grid[..., 0], mapped_grid[..., 1], mapped_grid[..., 2]



    #find the given trajectoires/predictions in the chart
    encode_traj = jax.vmap(model.psi, in_axes = 0)
    encode_trajs = jax.vmap(encode_traj, in_axes = 0)

    chart_trajectories = encode_trajs(data_trajectories)
    chart_predictions = encode_trajs(data_predictions)



    #plotting
    fig = plt.figure(figsize=(24, 12))

    #left: data trajectories and predictions
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.plot_wireframe(x_grid, y_grid, z_grid, color="gray", label=r'$\phi(U^P)$', alpha=0.4)


    #plot the data trajectories with initial point in black
    ax1.scatter(data_trajectories[:,0,0], data_trajectories[:,0,1], data_trajectories[:,0,2], color='black', marker = 'o', s = 20, label='data')
    ax1.quiver(data_trajectories[:,0,0], data_trajectories[:,0,1], data_trajectories[:,0,2],
               data_trajectories[:,0,3], data_trajectories[:,0,4], data_trajectories[:,0,5],
               color='black', length = 0.25)

    for i in range(data_trajectories.shape[0]):
        ax1.plot(data_trajectories[i, :, 0], data_trajectories[i, :, 1], data_trajectories[i, :, 2], color='black')

    #plot the data trajectories with initial point in red
    ax1.scatter(data_predictions[:,0,0], data_predictions[:,0,1], data_predictions[:,0,2], color='red', marker = 'o', s = 20, label='predictions')
    ax1.quiver(data_predictions[:,0,0], data_predictions[:,0,1], data_predictions[:,0,2],
               data_predictions[:,0,3], data_predictions[:,0,4], data_predictions[:,0,5],
               color='red', length = 0.25)

    for i in range(data_predictions.shape[0]):
        ax1.plot(data_predictions[i, :, 0], data_predictions[i, :, 1], data_predictions[i, :, 2], color='red')

    ax1.set_xlabel(r'$y^1$')
    ax1.set_ylabel(r'$y^2$')
    ax1.set_zlabel(r'$y^3$')
    ax1.set_title(r'flow until $t = $' + f'{time}' + r' in the data space $\tilde{N}$')
    ax1.legend()
    ax1.axis('equal')

    #right: chart inputs and targets / predictions
    ax2 = fig.add_subplot(122)

    # Horizontal grid lines
    ax2.plot(grid_x0.T, grid_x1.T, color='gray', linewidth=0.5)

    # Vertical grid lines
    ax2.plot(grid_x0, grid_x1, color='gray', linewidth=0.5)

    #plot the convex hull of the chart points
    ax2.plot(jnp.append(chart_data_points[chart_hull.vertices, 0], chart_data_points[chart_hull.vertices[0], 0]),
             jnp.append(chart_data_points[chart_hull.vertices, 1], chart_data_points[chart_hull.vertices[0], 1]),
             color='black', label = r'$U^P$', lw = 1)

    #plot the charted data trajectories with intial point in red
    ax2.scatter(chart_trajectories[:,0,0], chart_trajectories[:,0,1], color='black', marker = 'o', s = 20, label='encoded data')
    ax2.quiver(chart_trajectories[:,0,0], chart_trajectories[:,0,1],
               chart_trajectories[:,0,2], chart_trajectories[:,0,3],
               color='black', scale=7, scale_units="xy", width=0.003)
    for i in range(chart_trajectories.shape[0]):
        ax2.plot(chart_trajectories[i, :, 0], chart_trajectories[i, :, 1], color='black')

    #plot the charted data trajectories with intial point in red
    ax2.scatter(chart_predictions[:,0,0], chart_predictions[:,0,1], color='red', marker = 'o', s = 20, label='predictions')
    ax2.quiver(chart_predictions[:,0,0], chart_predictions[:,0,1],
               chart_predictions[:,0,2], chart_predictions[:,0,3],
               color='red', scale=7, scale_units="xy", width=0.003)
    for i in range(chart_predictions.shape[0]):
        ax2.plot(chart_predictions[i, :, 0], chart_predictions[i, :, 1], color='red')

    ax2.set_xlabel(r'$x^1$')
    ax2.set_ylabel(r'$x^2$')
    ax2.set_title(r'flow until $t = $' + f'{time}' + r' in the chart of the latent space $N$')
    ax2.legend()
    ax2.axis('equal')

    #adjust layout and display the plot
    plt.tight_layout()
    plt.show()

#auxilary function for the method below
def show_sectional_curvature(model, data_points):

    ######## data preparation ########

    #map the data points to the chart, shape (many, 4), and take only U^P shape (many, 2)
    chart_data_points = jax.vmap(model.psi, in_axes = 0)(data_points)[:,0:2]

    #find the chart domain as a convex hull
    chart_hull = ConvexHull(chart_data_points)

    #create an evaluation grid in the chart
    grid_res = (500,500)
    x0_vals = jnp.linspace(chart_data_points[:, 0].min(), chart_data_points[:, 0].max(), grid_res[0])
    x1_vals = jnp.linspace(chart_data_points[:, 1].min(), chart_data_points[:, 1].max(), grid_res[1])
    eval_grid = jnp.array(jnp.meshgrid(x0_vals, x1_vals)).T.reshape(-1, 2)


    ######## geometry ########

    #compute the sectional curvature on the finer evaluation grid
    curvature = jax.vmap(model.sectional_curvature, in_axes=0)(eval_grid)
    #reshape to match the grid resolution for plotting
    curvature = curvature.reshape(grid_res)


    ######## plotting #######
    fig = plt.figure(figsize=(12, 12))

    #charted data points with sectional curvature heatplot as background
    ax1 = fig.add_subplot(111)

    #plot heatmap of the sectional curvature
    im = ax1.imshow(curvature.T, extent=(x0_vals.min(), x0_vals.max(), x1_vals.min(), x1_vals.max()),
                    origin='lower', cmap='gist_rainbow')
    plt.colorbar(im, ax=ax1, label=r'sectional curvature')

    #plot the convex hull of the chart points
    ax1.plot(jnp.append(chart_data_points[chart_hull.vertices, 0], chart_data_points[chart_hull.vertices[0], 0]),
             jnp.append(chart_data_points[chart_hull.vertices, 1], chart_data_points[chart_hull.vertices[0], 1]),
             color='black', label = r'$U^P$', lw = 1)

    ax1.set_xlabel(r'$x^1$')
    ax1.set_ylabel(r'$x^2$')
    ax1.set_title(r'curvature of the learnt metric in the learnt chart $U^P$')
    ax1.legend()
    ax1.axis('equal')

    #adjust layout and display the plot
    plt.tight_layout()
    plt.show()

#visualizations shown for the toy problem
def toy_problem_visualizations(model_name, psi_initializer, phi_initializer, g_initializer):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)


    #load the data for prediction
    data_pred, _ = load_dataset(name = "half-sphere_trajectories_test",
                 size=50,
                 random_selection=True,
                 key=jax.random.PRNGKey(20))

    trajectories_pred, times_pred = data_pred

    #load the data for extrapolation
    data_extrapol, _ = load_dataset(name = "half-sphere_long-trajectories_test",
                 size=10,
                 random_selection=True,
                 key=jax.random.PRNGKey(20))

    trajectories_extrapol, times_extrapol = data_extrapol

    #load the data for curvature
    data_curv, _ = load_dataset(name = "half-sphere_trajectories_test",
             size=1024,
             random_selection=True,
             key=jax.random.PRNGKey(20))

    points_curv, _ = data_curv

    points_curv = points_curv.reshape(-1, points_curv.shape[-1]) #from (many, timesteps, ...) to (many*timesteps, ...)


    #calculate predictions
    pred = apply_model_function(model.get_geodesic,
                                tuple((trajectories_pred[:,0,:],
                                       times_pred[:,-1],
                                       times_pred.shape[1] - 1)),
                                vmap_axes = (0,0,None))

    #calculate extrapolated predictions
    extrapol = apply_model_function(model.get_geodesic,
                                    tuple((trajectories_extrapol[:,0,:],
                                           times_extrapol[:,-1],
                                           times_extrapol.shape[1] - 1)),
                                    vmap_axes = (0,0,None))

    #show predictions
    show_trajectories(model, trajectories_pred, pred, time = 1)

    #show time extrapolation
    show_trajectories(model, trajectories_extrapol, extrapol, time = 2)

    #calculate and show sectional curvature
    show_sectional_curvature(model, points_curv)


################################ Inference of model A ################################

if show_model_A:

    model_name = "master_thesis/toy-problem_model-A"

    psi_initializer = NN_Jacobian_split_diffeomorphism
    phi_initializer = NN_Jacobian_split_diffeomorphism
    g_initializer = NN_metric

    toy_problem_error_analysis(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

    toy_problem_visualizations(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

################################ Inference of model B ################################

if show_model_B:

    model_name = "master_thesis/toy-problem_model-B"

    psi_initializer = NN_diffeomorphism
    phi_initializer = NN_diffeomorphism
    g_initializer = NN_metric

    toy_problem_error_analysis(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

    toy_problem_visualizations(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

################################ Inference of model C ################################

if show_model_C:

    model_name = "master_thesis/toy-problem_model-C"

    psi_initializer = NN_Jacobian_split_diffeomorphism
    phi_initializer = NN_Jacobian_split_diffeomorphism
    g_initializer = NN_metric

    toy_problem_error_analysis(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

    toy_problem_visualizations(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

################################ Inference of model D ################################

if show_model_D:

    model_name = "master_thesis/toy-problem_model-D"

    psi_initializer = NN_diffeomorphism
    phi_initializer = NN_diffeomorphism
    g_initializer = NN_metric

    toy_problem_error_analysis(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)

    toy_problem_visualizations(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer)
