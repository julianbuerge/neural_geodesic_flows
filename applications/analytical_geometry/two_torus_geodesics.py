"""
Computation and visualization of geodesics on the two torus.
"""

import jax
import jax.numpy as jnp

import numpy as np

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_T2,
    psi_T2,
    phi_T2,
    g_T2
)

from core.models import (
    TangentBundle
)

from applications.analytical_geometry.visualizations import (
    chart_geodesic_visualization,
    parametrized_geodesic_visualization
)

torusbundle = TangentBundle(dim_M = 2, dim_dataspace = 6, psi = psi_T2, phi = phi_T2, g = g_T2)

#initial point in the chart (consists of initial x^0, x^1, v^0, v^1) (0 small circle, 1 big circle)
initial_point = jnp.array([0.5, 0, 1, 1])

#integration time
t = 5

#integration steps
steps = 100

#integration
chart_geodesic = torusbundle.exp_return_trajectory(initial_point, t, steps)

geodesic = jax.vmap(torusbundle.phi, in_axes = 0)(chart_geodesic)

#visualization
chart_geodesic_visualization(torusbundle, chart_geodesic, chartdomain_T2, name = 'torus')
parametrized_geodesic_visualization(torusbundle, geodesic, chartdomain_T2, name = 'torus')
