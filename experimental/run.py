"""
Test everything that we've implemented so far:

- create a multi chart tangent bundle for the two sphere
- calculate some geodesic on it with chart switching and plot it
"""

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
    psi_S2_spherical,
    phi_S2_spherical,
    g_S2_spherical
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
    create_coordinate_domains,
    Chart,
)

from experimental.tangent_bundle import (
    TangetBundle_multi_chart_atlas as TangetBundle,
)

### load the sphere data ###
size = 512
data, _ = load_dataset(name = "sphere_trajectories_train", size = size)

trajectories, times = data

### create 2 coordinate domains ###

#extract and flatten positions (x, y, z)
sphere = trajectories[..., 0:3].reshape(-1, 3)  # shape (size*time, 3)

#apply masks to get coordinate domains
extended_upper_hemisphere, extended_lower_hemisphere = create_coordinate_domains(sphere, k = 2, extension_degree = 0)

#initialize chart eqx.modules
psi_extended_upper_hemisphere = psi_S2_spherical#psi_S2_inverted_stereographic
phi_extended_upper_hemisphere = phi_S2_spherical#phi_S2_inverted_stereographic
g_extended_upper_hemisphere = g_S2_spherical#g_S2_stereographic#identity_metric({'dim_M':2})

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
initial_point = jnp.array([0.5, -0.4, -0.9, 0.3])
chart_id = 1

initial_state = (chart_id, initial_point)


#integration time
t = 8

#integration steps
steps = 250

#visualize the geodesic in data space and in the charts
sphere = parametrized_surface(parametrization_S2_spherical, chartdomain_S2_spherical)

full_dynamics_visualization(sphere_bundle, initial_state, t = t, steps = steps, surface = sphere)
