"""
Computation and visualization of geodesics on the two sphere.
"""

import jax
import jax.numpy as jnp

import numpy as np

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_S2_spherical,
    psi_S2_spherical,
    phi_S2_spherical,
    g_S2_spherical
)

from core.models import (
    TangentBundle
)

from applications.analytical_geometry.visualizations import (
    chart_geodesic_visualization,
    parametrized_geodesic_visualization
)

spherebundle = TangentBundle(dim_M = 2, dim_dataspace = 6, psi = psi_S2_spherical, phi = phi_S2_spherical, g = g_S2_spherical)

#initial point in the chart (consists of initial theta, phi, v^theta, v^phi)
initial_point = jnp.array([0.5, 0, 1, 1])

#integration time
t = 5

#integration steps
steps = 100

#integration
chart_geodesic = spherebundle.exp_return_trajectory(initial_point, t, steps)

geodesic = jax.vmap(spherebundle.phi, in_axes = 0)(chart_geodesic)

#visualization
chart_geodesic_visualization(spherebundle, chart_geodesic, chartdomain_S2_spherical, name = 'sphere')
parametrized_geodesic_visualization(spherebundle, geodesic, chartdomain_S2_spherical, name = 'sphere')
