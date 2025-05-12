"""
Computation and visualization of geodesics on the statistical manifold of one dimensional Gaussians,
equipped with the Fisher information matrix as the Riemannian metric.
"""

import jax
import jax.numpy as jnp

import numpy as np

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_Gaussians1,
    psi_Gaussians1,
    phi_Gaussians1,
    g_Gaussians1
)

from core.models import (
    TangentBundle
)

from applications.analytical_geometry.visualizations import (
    chart_geodesic_visualization
)

gaussiansbundle = TangentBundle(dim_M = 2, dim_dataspace = 6, psi = psi_Gaussians1, phi = phi_Gaussians1, g = g_Gaussians1)

#initial point in the chart (consists of initial mu, sigma, v^mu, v^sigma)
initial_point = jnp.array([0, 1.0, 1, 1])

#integration time
t = 10

#integration steps
steps = 100

#integration
chart_geodesic = gaussiansbundle.exp_return_trajectory(initial_point, t, steps)


#visualization
chart_geodesic_visualization(gaussiansbundle, chart_geodesic, chartdomain_Gaussians1, name = 'Gaussians manifold')
