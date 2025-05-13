"""
Replication of the case study "Two body problem" inference
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
from core.template_psi_phi_g_functions_analytical import (
    two_body_Jacobi_metric
)
from core.template_psi_phi_g_functions_neural_networks import (
    NN_diffeomorphism_for_chart,
    NN_diffeomorphism,
    NN_Jacobian_split_diffeomorphism_for_chart,
    NN_Jacobian_split_diffeomorphism,
    NN_metric_regularized,
)

from core.models import (
    TangentBundle
)

from core.inference import (
    apply_model_function,
    find_indices,
    trajectory_model_analyis
)

#get some loading methods to load models and test datasets
from applications.utils import (
    load_dataset,
    load_model,
)

#choose what models inference to show
show_model_A = False    #fixed energy model
show_model_B = False    #fixed energy model
show_model_C = False    #model does not solve learning task
show_model_D = True     #best model, varying energy


############################# numerical inference methods #############################

def two_body_problem_error_analysis(model_name,
                               psi_initializer,
                               phi_initializer,
                               g_initializer,
                               energy):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)

    #check if we are in the fixed or varying energy case
    if energy == "fixed":
        dataset = "two-body-problem_fixed-energy-trajectories_test"
    else:
        dataset = "two-body-problem_trajectories_test"

    #load the data, respecting the correct mode
    data, _ = load_dataset(name = dataset,
                 size=900, #varying energy will load 900 rabdomly selected, fixed energy will load all 15
                 random_selection=True,
                 key=jax.random.PRNGKey(20))

    trajectories, times = data

    trajectory_model_analyis(model, trajectories, times)

def two_body_problem_energy_analysis(model_name,
                                     psi_initializer,
                                     phi_initializer,
                                     g_initializer,
                                     energy):

        #load the model
        model = load_model(model_name,
                           psi_initializer = psi_initializer,
                           phi_initializer = phi_initializer,
                           g_initializer = g_initializer)

        #check if we are in the fixed or varying energy case
        if energy == "fixed":
            dataset = "two-body-problem_fixed-energy-trajectories_test"
        else:
            dataset = "two-body-problem_trajectories_test"

        #load the data, respecting the correct mode
        data, _ = load_dataset(name = dataset,
                     size=900, #varying energy will load 900 rabdomly selected, fixed energy will load all 15
                     random_selection=True,
                     key=jax.random.PRNGKey(20))

        data_trajectories, times = data

        ######## geometry ########

        #vmap the functions over one trajectory and then several trajectories
        encode_traj = jax.vmap(model.psi, in_axes = 0)
        encode_trajs = jax.vmap(encode_traj, in_axes = 0)

        geo = jax.vmap(model.exp_return_trajectory, in_axes = (0,0,None))

        decode_geo = jax.vmap(model.phi, in_axes = 0)
        decode_geos = jax.vmap(decode_geo, in_axes = 0)

        #apply the functions
        chart_trajectories = encode_trajs(data_trajectories)

        #start each prediction at the first point of the timesteps
        chart_predictions = geo(chart_trajectories[:,0,:], times[:,-1], times.shape[1]-1)

        data_predictions = decode_geos(chart_predictions)

        ######## energy functions ########

        #in the data space
        potential_energy = lambda yx : -1/jnp.linalg.norm(yx[0:2]-yx[2:4])
        kinetic_energy = lambda yv : 0.5*jnp.sum(yv**2)

        total_energy = lambda y : potential_energy(y[0:4]) + kinetic_energy(y[4:8])

        #in the chart
        geo_energy = lambda z : model.scalarproduct(z[0:2],z[2:4],z[2:4])

        #vmap along an orbit
        potential_energy_orbit = jax.vmap(potential_energy, in_axes = 0)
        kinetic_energy_orbit = jax.vmap(kinetic_energy, in_axes = 0)
        total_energy_orbit = jax.vmap(total_energy, in_axes = 0)
        geo_energy_orbit = jax.vmap(geo_energy, in_axes = 0)

        #vmap along orbits
        potential_energy_orbits = jax.vmap(potential_energy_orbit, in_axes = 0)
        kinetic_energy_orbits = jax.vmap(kinetic_energy_orbit, in_axes = 0)
        total_energy_orbits = jax.vmap(total_energy_orbit, in_axes = 0)
        geo_energy_orbits = jax.vmap(geo_energy_orbit, in_axes = 0)

        ######## energy calculations ########
        trajectories_potential_energy = potential_energy_orbits(data_trajectories[:,:,0:4])
        trajectories_kinetic_energy = kinetic_energy_orbits(data_trajectories[:,:,4:8])
        trajectories_total_energy = total_energy_orbits(data_trajectories)
        trajectories_geo_energy = geo_energy_orbits(chart_trajectories)

        predictions_potential_energy = potential_energy_orbits(data_predictions[:,:,0:4])
        predictions_kinetic_energy = kinetic_energy_orbits(data_predictions[:,:,4:8])
        predictions_total_energy = total_energy_orbits(data_predictions)
        predictions_geo_energy = geo_energy_orbits(chart_predictions)



        ######## Load HNN Baseline ########

        from applications.master_thesis.HNN.get_HNN_prediction import (
            trained_HNN_prediction
        )

        hnn_trajectories = jnp.zeros_like(data_trajectories)

        #since hnn is not made with jax we loop through all
        for i in range(data_trajectories.shape[0]):

            #expect to be of shape (2,5, timesteps) where it is 2 bodies with (mass,x,y,vx,vy)
            hnn_orbit = trained_HNN_prediction(data_trajectories[i,0,:], time_steps = times.shape[1])

            #turn into an array of shape (timesteps, 8) just like my coordinates: (xA,yA,xB,yB,vxA,vyA,vxB,vyB)
            body_yxA = jnp.transpose(hnn_orbit[0, 1:3, :])
            body_yvA = jnp.transpose(hnn_orbit[0, 3:5, :])
            body_yxB = jnp.transpose(hnn_orbit[1, 1:3, :])
            body_yvB = jnp.transpose(hnn_orbit[1, 3:5, :])
            hnn_trajectory = jnp.concatenate([body_yxA, body_yxB, body_yvA, body_yvB], axis=1)

            hnn_trajectories = hnn_trajectories.at[i,:,:].set(hnn_trajectory)


        #find its energies
        hnn_potential_energy = potential_energy_orbits(hnn_trajectories[:,0:4])
        hnn_kinetic_energy = kinetic_energy_orbits(hnn_trajectories[:,4:8])
        hnn_total_energy = total_energy_orbits(hnn_trajectories)



        ######## print results ########

        #errors
        error_total_energy_predicted = jnp.mean((predictions_total_energy - trajectories_total_energy)**2)

        error_total_energy_hnn = jnp.mean((hnn_total_energy - trajectories_total_energy)**2)

        #mean deviation in geodesic energy
        trajectories_var = jnp.var(trajectories_geo_energy)
        trajectories_md = jnp.std(trajectories_geo_energy)/jnp.mean(trajectories_geo_energy)

        predictions_var = jnp.var(predictions_geo_energy)
        predictions_md = jnp.std(predictions_geo_energy)/jnp.mean(predictions_geo_energy)


        print(f"\n\nMSE of the total energy as predicted by the model {error_total_energy_predicted}")

        print(f"\nMSE of the total energy as predicted by HNN {error_total_energy_hnn}")

        print(f"\n\nVariance of geodesic energy of data {trajectories_var}")
        print(f"Mean deviation of geodesic energy of data {trajectories_md}")
        print(f"\nVariance of geodesic energy of predicted {predictions_var}")
        print(f"Mean deviation of geodesic energy of predicted {predictions_md}\n\n")



############################# auxilary methods for the visualizations #############################

def show_trajectories(model,
                      data_trajectories, #given trajectories in the data space
                      data_predictions, #predictions in the data space
                      indices, case):

    ######## data preparation ########

    #collect all points together to visualize the chart domain
    data_points = data_trajectories.reshape(-1, data_trajectories.shape[-1]) #from (many, timesteps, ...) to (many*timesteps, ...)

    #map the data points to the chart, shape (many, 4)
    chart_data_points = jax.vmap(model.psi, in_axes = 0)(data_points)

    #find the chart domain of only U^P shape (many, 2) as a convex hull
    chart_hull = ConvexHull(chart_data_points[:,0:2])

    #select only the chosen ones for doing the forward
    indices = jnp.array(indices)

    data_trajectories = data_trajectories[indices]
    data_predictions = data_predictions[indices]


    ######## geometry ########

    encode_traj = jax.vmap(model.psi, in_axes = 0)
    encode_trajs = jax.vmap(encode_traj, in_axes = 0)

    chart_trajectories = encode_trajs(data_trajectories)
    chart_predictions = encode_trajs(data_predictions)



    ######## plotting #######
    fig = plt.figure(figsize=(24, 12))

    #left: data trajectories and predictions
    ax1 = fig.add_subplot(121)

    #data body one
    ax1.scatter(data_trajectories[:,0,0],data_trajectories[:,0,1],color='black',label='data body one',marker='o',s=100)

    ax1.quiver(data_trajectories[:,0,0],data_trajectories[:,0,1],
                    data_trajectories[:,0,4],data_trajectories[:,0,5],
                        width = 0.0025,color='black')

    for i in range(data_trajectories.shape[0]):
        ax1.plot(data_trajectories[i,:,0],data_trajectories[i,:,1],color='black')

    #data body two
    ax1.scatter(data_trajectories[:,0,2],data_trajectories[:,0,3],color='black',label='data body two',marker='s',s=100)

    ax1.quiver(data_trajectories[:,0,2],data_trajectories[:,0,3],
                    data_trajectories[:,0,6],data_trajectories[:,0,7],
                        width = 0.0025,color='black')

    for i in range(data_trajectories.shape[0]):
        ax1.plot(data_trajectories[i,:,2],data_trajectories[i,:,3],color='black')


    #prediction body one
    ax1.scatter(data_predictions[:,0,0],data_predictions[:,0,1],color='red',label='prediction body one',marker='+',s=100)

    ax1.quiver(data_predictions[:,0,0],data_predictions[:,0,1],
                    data_predictions[:,0,4],data_predictions[:,0,5],
                        width = 0.0025,color='red')

    for i in range(data_predictions.shape[0]):
        ax1.plot(data_predictions[i,:,0],data_predictions[i,:,1],color='red')

    #prediction body two
    ax1.scatter(data_predictions[:,0,2],data_predictions[:,0,3],color='red',label='prediction body two',marker='x',s=100)

    ax1.quiver(data_predictions[:,0,2],data_predictions[:,0,3],
                    data_predictions[:,0,6],data_predictions[:,0,7],
                        width = 0.0025,color='red')

    for i in range(data_trajectories.shape[0]):
        ax1.plot(data_predictions[i,:,2],data_predictions[i,:,3],color='red')

    ax1.set_xlabel(r'$y^1$')
    ax1.set_ylabel(r'$y^2$')
    ax1.set_title(f'{case}' + r' flow in the data space $\tilde{N}$')
    ax1.legend()
    ax1.axis('equal')

    #right: chart inputs and targets / predictions
    ax2 = fig.add_subplot(122)

    #plot the convex hull of the chart points
    ax2.plot(jnp.append(chart_data_points[chart_hull.vertices, 0], chart_data_points[chart_hull.vertices[0], 0]),
             jnp.append(chart_data_points[chart_hull.vertices, 1], chart_data_points[chart_hull.vertices[0], 1]),
             color='black', label = r'$U^P$', lw = 1)

    #plot the charted data trajectories with intial point in red
    ax2.scatter(chart_trajectories[:,0,0], chart_trajectories[:,0,1], color='black', marker = 'o', s = 20, label='encoded data')
    ax2.quiver(chart_trajectories[:,0,0], chart_trajectories[:,0,1],
               chart_trajectories[:,0,2], chart_trajectories[:,0,3],
               color='black', scale=3, scale_units="xy", width=0.003)
    for i in range(indices.shape[0]):
        ax2.plot(chart_trajectories[i, :, 0], chart_trajectories[i, :, 1], color='black')


    #plot the charted prediction trajectories with intial point in red
    ax2.scatter(chart_predictions[:,0,0], chart_predictions[:,0,1], color='red', marker = 'o', s = 20, label='predictions')
    ax2.quiver(chart_predictions[:,0,0], chart_predictions[:,0,1],
               chart_predictions[:,0,2], chart_predictions[:,0,3],
               color='red', scale=3, scale_units="xy", width=0.003)
    for i in range(indices.shape[0]):
        ax2.plot(chart_predictions[i, :, 0], chart_predictions[i, :, 1], color='red')

    ax2.set_xlabel(r'$x^1$')
    ax2.set_ylabel(r'$x^2$')
    ax2.set_title(f'{case}' + r' flow in the chart of the latent space $N$')
    ax2.legend()
    ax2.axis('equal')

    #adjust layout and display the plot
    plt.tight_layout()
    plt.show()

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

def show_Jacobi_curvature(model, data_points, total_energy):

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

    #pullback the Jacobi metric to the chart of tangentbundle
    g_Jacobi = two_body_Jacobi_metric(model.phi, {'total_energy':total_energy})

    #build a new tangentbundle with that metric
    tangentbundle = TangentBundle(dim_dataspace = 8, dim_M = 2,
                                  psi = model.psi,
                                  phi = model.phi,
                                  g = g_Jacobi)

    #compute the sectional curvature on the finer evaluation grid
    curvature = jax.vmap(tangentbundle.sectional_curvature, in_axes=0)(eval_grid)
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
    ax1.set_title(r'curvature of the Jacobi metric pulled back to the learnt chart $U^P$')
    ax1.legend()
    ax1.axis('equal')

    #adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def show_energies(model,
                  data_trajectories, #given trajectories in the data space
                  data_predictions, #predictions in the data space
                  times,
                  index, case):

    ######## data preparation ########

    #select only the chosen one
    data_trajectory = data_trajectories[index,...]
    data_prediction = data_predictions[index,...]
    time = times[index,...]


    ######## geometry ########

    encode_traj = jax.vmap(model.psi, in_axes = 0)

    decode_geo = jax.vmap(model.phi, in_axes = 0)

    chart_trajectory = encode_traj(data_trajectory)

    chart_prediction = encode_traj(data_prediction)


    ######## energy functions ########

    #in the data space
    potential_energy = lambda yx : -1/jnp.linalg.norm(yx[0:2]-yx[2:4])
    kinetic_energy = lambda yv : 0.5*jnp.sum(yv**2)

    total_energy = lambda y : potential_energy(y[0:4]) + kinetic_energy(y[4:8])

    #in the chart
    geo_energy = lambda z : model.scalarproduct(z[0:2],z[2:4],z[2:4])

    #vmap along an orbit
    potential_energy_orbit = jax.vmap(potential_energy, in_axes = 0)
    kinetic_energy_orbit = jax.vmap(kinetic_energy, in_axes = 0)
    total_energy_orbit = jax.vmap(total_energy, in_axes = 0)
    geo_energy_orbit = jax.vmap(geo_energy, in_axes = 0)

    ######## energy calculations ########
    trajectory_potential_energy = potential_energy_orbit(data_trajectory[:,0:4])
    trajectory_kinetic_energy = kinetic_energy_orbit(data_trajectory[:,4:8])
    trajectory_total_energy = total_energy_orbit(data_trajectory)
    trajectory_geo_energy = geo_energy_orbit(chart_trajectory)

    prediction_potential_energy = potential_energy_orbit(data_prediction[:,0:4])
    prediction_kinetic_energy = kinetic_energy_orbit(data_prediction[:,4:8])
    prediction_total_energy = total_energy_orbit(data_prediction)
    prediction_geo_energy = geo_energy_orbit(chart_prediction)



    ######## Load HNN Baseline ########

    from applications.master_thesis.HNN.get_HNN_prediction import (
        trained_HNN_prediction
    )

    #expect to be of shape (2,5, timesteps) where it is 2 bodies with (mass,x,y,vx,vy)
    hnn_orbit = trained_HNN_prediction(data_trajectory[0,:], time_steps = time.shape[0])

    #turn into an array of shape (timesteps, 8) just like my coordinates: (xA,yA,xB,yB,vxA,vyA,vxB,vyB)
    body_yxA = jnp.transpose(hnn_orbit[0, 1:3, :])
    body_yvA = jnp.transpose(hnn_orbit[0, 3:5, :])
    body_yxB = jnp.transpose(hnn_orbit[1, 1:3, :])
    body_yvB = jnp.transpose(hnn_orbit[1, 3:5, :])
    hnn_trajectory = jnp.concatenate([body_yxA, body_yxB, body_yvA, body_yvB], axis=1)

    hnn_chart_trajectory = encode_traj(hnn_trajectory)

    #find its energies
    hnn_potential_energy = potential_energy_orbit(hnn_trajectory[:,0:4])
    hnn_kinetic_energy = kinetic_energy_orbit(hnn_trajectory[:,4:8])
    hnn_total_energy = total_energy_orbit(hnn_trajectory)


    ######## plotting #######
    fig = plt.figure(figsize=(24, 12))

    #left: data trajectory and prediction
    ax1 = fig.add_subplot(121)

    #data body one
    ax1.scatter(data_trajectory[0,0],data_trajectory[0,1],color='black',label='data body one',marker='o',s=100)

    ax1.quiver(data_trajectory[0,0],data_trajectory[0,1],
                    data_trajectory[0,4],data_trajectory[0,5],
                        width = 0.0025,color='black')

    ax1.plot(data_trajectory[:,0],data_trajectory[:,1],color='black')

    #data body two
    ax1.scatter(data_trajectory[0,2],data_trajectory[0,3],color='black',label='data body two',marker='s',s=100)

    ax1.quiver(data_trajectory[0,2],data_trajectory[0,3],
                    data_trajectory[0,6],data_trajectory[0,7],
                        width = 0.0025,color='black')

    ax1.plot(data_trajectory[:,2],data_trajectory[:,3],color='black')

    #HNN prediction
    ax1.plot(hnn_trajectory[:,0],hnn_trajectory[:,1],color='gray', label = 'HNN prediction body one', marker = 'p')
    ax1.plot(hnn_trajectory[:,2],hnn_trajectory[:,3],color='gray', label = 'HNN prediction body two', marker = '*')

    #prediction body one
    ax1.scatter(data_prediction[0,0],data_prediction[0,1],color='red',label='prediction body one',marker='+',s=100)

    ax1.quiver(data_prediction[0,0],data_prediction[0,1],
                    data_prediction[0,4],data_prediction[0,5],
                        width = 0.0025,color='red')

    ax1.plot(data_prediction[:,0],data_prediction[:,1],color='red')

    #prediction body two
    ax1.scatter(data_prediction[0,2],data_prediction[0,3],color='red',label='prediction body two',marker='x',s=100)

    ax1.quiver(data_prediction[0,2],data_prediction[0,3],
                    data_prediction[0,6],data_prediction[0,7],
                        width = 0.0025,color='red')

    ax1.plot(data_prediction[:,2],data_prediction[:,3],color='red')

    ax1.set_xlabel(r'$y^1$')
    ax1.set_ylabel(r'$y^2$')
    ax1.set_title(r'flow in the data space $\tilde{N}$')
    ax1.legend()
    ax1.axis('equal')

    #right: energies
    ax2 = fig.add_subplot(122)

    ax2.plot(time, trajectory_potential_energy, color = 'orange', label = 'potential of data')
    ax2.plot(time, trajectory_kinetic_energy, color = 'blue', label = 'kinetic of data')
    ax2.plot(time, trajectory_total_energy, color = 'black', label = 'total of data')
    ax2.plot(time, trajectory_geo_energy, color = 'gray', label = 'geodesic of data')

    ax2.plot(time, hnn_potential_energy, color = 'orange', label = 'potential of HNN', marker = '^')
    ax2.plot(time, hnn_kinetic_energy, color = 'blue', label = 'kinetic of HNN', marker = '^')
    ax2.plot(time, hnn_total_energy, color = 'black', label = 'total of HNN', marker = '^')
    #ax2.plot(time, hnn_geo_energy, color = 'gray', label = 'geodesic of HNN', marker = '^')

    ax2.plot(jnp.linspace(0,time[-1],times.shape[1]), prediction_potential_energy, color = 'orange', label = 'potential of predicted', marker = 'x')
    ax2.plot(jnp.linspace(0,time[-1],times.shape[1]), prediction_kinetic_energy, color = 'blue', label = 'kinetic of predicted', marker = 'x')
    ax2.plot(jnp.linspace(0,time[-1],times.shape[1]), prediction_total_energy, color = 'black', label = 'total of predicted', marker = 'x')
    ax2.plot(jnp.linspace(0,time[-1],times.shape[1]), prediction_geo_energy, color = 'gray', label = 'geodesic of predicted', marker = 'x')

    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$E$')
    ax2.set_title(r'Energy values on the trajectory')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax2.axis('equal')

    #adjust layout and display the plot
    plt.tight_layout()
    plt.show()





############################# visual inference methods #############################

def two_body_problem_visualizations(model_name, psi_initializer, phi_initializer, g_initializer, energy):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)

    #check if we are in the fixed or varying energy case
    if energy == "fixed":
        dataset = "two-body-problem_fixed-energy-trajectories_test"
    else:
        dataset = "two-body-problem_trajectories_test"

    #load the data for prediction
    data_pred, _ = load_dataset(name = dataset,
                 size=900, #varying energy will load 900 rabdomly selected, fixed energy will load all 15
                 random_selection=True,
                 key=jax.random.PRNGKey(20))

    trajectories_pred, times_pred = data_pred

    #load the data for curvature, which is for fixed energy in all cases (to compare with g_jacobi_h=-0.153)
    data_curv, _ = load_dataset(name = "two-body-problem_fixed-energy-trajectories_test")

    points_curv, _ = data_curv

    points_curv = points_curv.reshape(-1, points_curv.shape[-1]) #from (many, timesteps, ...) to (many*timesteps, ...)


    #calculate predictions
    pred = apply_model_function(model.get_geodesic,
                                tuple((trajectories_pred[:,0,:],
                                       times_pred[:,-1],
                                       times_pred.shape[1] - 1)),
                                vmap_axes = (0,0,None))

    #check if we are in the fixed or varying energy case
    if energy == "fixed":

        #show all 15 test trajectories
        show_trajectories(model, trajectories_pred, pred, indices = jnp.arange(15), case = "entire fixed energy test")

    else:
        #find the worst and average predictions
        ind_wrst, ind_avg, _ = find_indices(correct_outputs = trajectories_pred,
                                            model_outputs = pred,
                                            size = 15)

        #show the average predictions
        show_trajectories(model, trajectories_pred, pred, indices = ind_avg, case = "average varying energy test")

        #show the average predictions
        show_trajectories(model, trajectories_pred, pred, indices = ind_wrst, case = "worst varying energy test")


    #calculate and show sectional curvature
    show_sectional_curvature(model, points_curv)
    show_Jacobi_curvature(model, points_curv, total_energy = -0.153)

def two_body_problem_energy_visualizations(model_name, psi_initializer, phi_initializer, g_initializer, energy):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)

    #check if we are in the fixed or varying energy case
    if energy == "fixed":
        dataset = "two-body-problem_fixed-energy-trajectories_test"
    else:
        dataset = "two-body-problem_trajectories_test"

    #load the data for prediction
    data_pred, _ = load_dataset(name = dataset,
                 size=900, #varying energy will load 900 rabdomly selected, fixed energy will load all 15
                 random_selection=True,
                 key=jax.random.PRNGKey(20))

    trajectories_pred, times_pred = data_pred


    #calculate predictions
    pred = apply_model_function(model.get_geodesic,
                                tuple((trajectories_pred[:,0,:],
                                       times_pred[:,-1],
                                       times_pred.shape[1] - 1)),
                                vmap_axes = (0,0,None))

    #check if we are in the fixed or varying energy case
    if energy == "fixed":
        show_energies(model, trajectories_pred, pred, times_pred, index = 5, case = "")
    else:
        show_energies(model, trajectories_pred, pred, times_pred, index = 300, case = "")


################################ Inference of model A ################################

if show_model_A:

    model_name = "master_thesis/two-body-problem_model-A"

    psi_initializer = NN_Jacobian_split_diffeomorphism_for_chart
    phi_initializer = NN_Jacobian_split_diffeomorphism
    g_initializer = NN_metric_regularized

    two_body_problem_error_analysis(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "fixed")

    two_body_problem_visualizations(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "fixed")

    two_body_problem_energy_analysis(model_name,
                                     psi_initializer,
                                     phi_initializer,
                                     g_initializer,
                                     energy = "fixed")

    two_body_problem_energy_visualizations(model_name,
                                           psi_initializer,
                                           phi_initializer,
                                           g_initializer,
                                           energy = "fixed")

################################ Inference of model B ################################

if show_model_B:

    model_name = "master_thesis/two-body-problem_model-B"

    psi_initializer = NN_diffeomorphism_for_chart
    phi_initializer = NN_diffeomorphism
    g_initializer = NN_metric_regularized

    two_body_problem_error_analysis(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "fixed")

    two_body_problem_visualizations(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "fixed")

    two_body_problem_energy_analysis(model_name,
                                     psi_initializer,
                                     phi_initializer,
                                     g_initializer,
                                     energy = "fixed")

    two_body_problem_energy_visualizations(model_name,
                                           psi_initializer,
                                           phi_initializer,
                                           g_initializer,
                                           energy = "fixed")

################################ Inference of model C ################################

if show_model_C:

    model_name = "master_thesis/two-body-problem_model-C"

    psi_initializer = NN_Jacobian_split_diffeomorphism_for_chart
    phi_initializer = NN_Jacobian_split_diffeomorphism
    g_initializer = NN_metric_regularized

    two_body_problem_error_analysis(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "varying")

    two_body_problem_visualizations(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "varying")

    two_body_problem_energy_analysis(model_name,
                                     psi_initializer,
                                     phi_initializer,
                                     g_initializer,
                                     energy = "varying")

    two_body_problem_energy_visualizations(model_name,
                                           psi_initializer,
                                           phi_initializer,
                                           g_initializer,
                                           energy = "varying")

################################ Inference of model D ################################

if show_model_D:

    model_name = "master_thesis/two-body-problem_model-D"

    psi_initializer = NN_diffeomorphism_for_chart
    phi_initializer = NN_diffeomorphism
    g_initializer = NN_metric_regularized

    two_body_problem_error_analysis(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "varying")

    two_body_problem_visualizations(model_name,
                                    psi_initializer,
                                    phi_initializer,
                                    g_initializer,
                                    energy = "varying")

    two_body_problem_energy_analysis(model_name,
                                     psi_initializer,
                                     phi_initializer,
                                     g_initializer,
                                     energy = "varying")

    two_body_problem_energy_visualizations(model_name,
                                           psi_initializer,
                                           phi_initializer,
                                           g_initializer,
                                           energy = "varying")
