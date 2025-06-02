import jax
import jax.numpy as jnp

import equinox as eqx

class TangetBundle(eqx.Module):

    #tuple of Charts
    atlas : tuple

    #id = index of the currently used chart
    current_chart_id : int

    def __init__(self, atlas):

        self.atlas = atlas

        self.current_chart_id = 0

    def update_chart_id(z):

        #find out which point z_out is closest to in the current charts coordinate domain
        current_chart = self.atlas[self.current_chart_id]

        y = current_chart.phi(z)

        idy = current_chart.coordinate_domain.find_closest_point_bfs(y)

        #store this in the current chart for the next round (the bfs method will automatically start there)
        current_chart.coordinate_domain.last_closest_idy = idy


        #find out if this point is an interior point of any other chart
        for i, chart in enumerate(self.atlas):

            if i != self.current_chart_id and idy in chart.coordinate_domain.interior_indices:

                #if so update z into the new chart
                z = chart.psi(y)

                #& set the current_chart_id to that chart's id
                self.current_chart_id = i

                break

        return z

    #global dynamics using chart transitions
    def __call__(self, y_in, t, num_steps : int):

        dt = t/num_steps

        s = 0

        chart = self.atlas[self.current_chart_id]

        z = chart.psi(y_in)

        while s < t:

            chart = self.atlas[self.current_chart_id]

            z = chart.exp_single_time_step(z, dt)

            z = self.update_chart(z)

            s += dt

        chart = self.atlas[self.current_chart_id]

        y_out = chart.phi(z)

        return y_out
