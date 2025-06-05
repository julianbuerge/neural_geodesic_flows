import jax
import jax.numpy as jnp

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_S2_spherical,
    parametrization_S2_spherical,
    chartdomain_S2_stereographic,
    psi_S2_inverted_stereographic,
    phi_S2_inverted_stereographic,
    psi_S2_stereographic,
    phi_S2_stereographic,
    g_S2_stereographic,

)

from core.template_psi_phi_g_functions_neural_networks import (
    identity_metric,
)

from applications.utils import (
    load_dataset,
)

from experimental.inference import (
    parametrized_surface,
    full_dynamics_visualization
)

from experimental.atlas import (
    Chart,
)

from experimental.tangent_bundle import (
    TangetBundle,
)

### load the sphere data ###
size = 512
data, _ = load_dataset(name = "sphere_trajectories_train", size = size)

trajectories, times = data

### manually create 2 coordinate domains ###

#extract and flatten positions (x, y, z)
positions = trajectories[..., 0:3].reshape(-1, 3)  # shape (size*time, 3)

z_vals = positions[:, 2]

#create boolean masks
mask_upper = z_vals > -0.2
mask_lower = z_vals < 0.2

#apply masks to get coordinate domains
extended_upper_hemisphere = positions[mask_upper]
extended_lower_hemisphere = positions[mask_lower]

#initialize chart eqx.modules
psi_extended_upper_hemisphere = psi_S2_inverted_stereographic
phi_extended_upper_hemisphere = phi_S2_inverted_stereographic
g_extended_upper_hemisphere = identity_metric({'dim_M':2})

psi_extended_lower_hemisphere = psi_S2_stereographic
phi_extended_lower_hemisphere = phi_S2_stereographic
g_extended_lower_hemisphere = g_S2_stereographic#identity_metric({'dim_M':2})



### assign chart functions to the 2 coordinate domains (these need to work on those domains!) ###
chart_upper_hemisphere = Chart(coordinate_domain = extended_upper_hemisphere,
                               psi = psi_extended_upper_hemisphere,
                               phi = phi_extended_upper_hemisphere,
                               g = g_extended_upper_hemisphere)

chart_lower_hemisphere = Chart(coordinate_domain = extended_lower_hemisphere,
                               psi = psi_extended_lower_hemisphere,
                               phi = phi_extended_lower_hemisphere,
                               g = g_extended_lower_hemisphere)

sphere_atlas = (chart_upper_hemisphere, chart_lower_hemisphere)

### build a spherebundle and test global dynamics ###
sphere_bundle = TangetBundle(atlas = sphere_atlas)

#initial point in the chart (consists of initial theta, phi, v^theta, v^phi)
initial_point = jnp.array([0.3, 0.8, -1.0, -0.5])
chart_id = 1

initial_state = (chart_id, initial_point)


#integration time
t = 8.0

#integration steps
steps = 100

#visualize the geodesic in data space and in the charts
sphere = parametrized_surface(parametrization_S2_spherical, chartdomain_S2_spherical)

full_dynamics_visualization(sphere_bundle, initial_state, t = t, steps = steps, surface = sphere)
