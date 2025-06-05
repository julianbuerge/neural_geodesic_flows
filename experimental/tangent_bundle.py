import jax
import jax.numpy as jnp

import equinox as eqx



class TangetBundle(eqx.Module):

    #tuple of Charts
    atlas : tuple

    amount_of_charts : int

    def __init__(self, atlas):

        self.atlas = atlas

        self.amount_of_charts = len(atlas)


    #return array of shape (self.amount_of_charts,) holding 1.0 in the chart_id index and 0 everywhere else
    #because chart_id is traced you can't use it as an index!
    def partition_of_unity(self, chart_id):

        return jax.nn.one_hot(chart_id, self.amount_of_charts)

    #apply a function in (or accossiated with) each chart (through the passed tuple of functions)
    #to the same inputs (which are a tuple of all required arguments)
    #and select only the one from the desired chart
    def apply_function(self, functions, chart_id, inputs):

        #obtain an array with all zeros except at index chart_id, there it is 1, shape (amount_of_charts,)
        partition_of_unity = self.partition_of_unity(chart_id)

        #apply func in all charts, yielding shape(amount_of_charts, math dim)
        output_in_all_charts = jnp.stack([func(*inputs) for func in functions])

        #find the only correct output by doing a dot product between the partition of unity and all outputs
        #yields shape (math dim,)
        output = jnp.dot(partition_of_unity, output_in_all_charts)

        return output


    #for an input in the dataspace find out which coordinate domain it belongs to
    def determine_chart(self, y):

        north_pole = jnp.array([0.0, 0.0, 1.0])
        south_pole = jnp.array([0.0, 0.0, -1.0])

        dist_north = jnp.sum((y[:3] - north_pole) ** 2)
        dist_south = jnp.sum((y[:3] - south_pole) ** 2)

        # Decide chart index: 0 if closer to north pole, 1 if closer to south pole
        chart_id = jnp.argmin(jnp.array([dist_north, dist_south]))  # shape (), value 0 or 1

        return chart_id

    #test if z should stay in the chart chart_id or if we need to switch to another
    #if so, return the new chart_id and z in the *new* chart
    #if not, return the same chart_id and z
    def update_chart(self, state):

        chart_id, z = state

        y = self.phi(state)

        new_chart_id = self.determine_chart(y)

        # Check whether chart switch is necessary
        switch = new_chart_id != chart_id

        # Decode to data space and re-encode into new chart if needed
        def switched_z():

            # encode in new chart (don't use the global psi as that would call the expensive determine_chart again)
            z = self.apply_function(functions = tuple(chart.psi for chart in self.atlas),
                                    chart_id = new_chart_id,
                                    inputs = (y,))

            return z

        new_z = jax.lax.cond(switch, switched_z, lambda : z)

        return (new_chart_id, new_z)


    #encoder
    def psi(self, y):

        chart_id = self.determine_chart(y)

        z = self.apply_function(functions = tuple(chart.psi for chart in self.atlas),
                                chart_id = chart_id,
                                inputs = (y,))

        return z

    #decoder
    def phi(self, state):

        chart_id, z = state

        y = self.apply_function(functions = tuple(chart.phi for chart in self.atlas),
                                chart_id = chart_id,
                                inputs = (z,))

        return y

    #metric
    def g(self, state):

        chart_id, z = state

        G = self.apply_function(functions = tuple(chart.g for chart in self.atlas),
                                chart_id = chart_id,
                                inputs = (z,))

        return G

    #geodesic evolution, state = (chart_id, z)
    def exp_return_trajectory(self, state, t, num_steps: int):

        dt = t / num_steps

        def step_function(state, _):

            #retrieve current chart and latent value
            chart_id, z = state

            z_next = self.apply_function(functions = tuple(chart.exp_single_time_step for chart in self.atlas),
                                         chart_id = chart_id,
                                         inputs = (z, dt))

            #possibly switch chart
            next_state = (chart_id, z_next)

            next_state = self.update_chart(next_state)

            return next_state, next_state

        initial_state = state

        #obtain tuple of two arrays, one is the chart_ids shape (num_steps,), the other the trajectory shape (num_steps, math dim)
        _, geodesic_evolution = jax.lax.scan(step_function, initial_state, None, length=num_steps)

        chart_ids_evolution, zs_evolution = geodesic_evolution

        #obtain the two arrays from the initial state
        chart_id_initial, z_initial = initial_state

        chart_id_initial = jnp.expand_dims(chart_id_initial, axis=0)
        z_initial = jnp.expand_dims(z_initial, axis=0)


        #prepend the initial state to obtain the full geodesic (initial & evolution)
        chart_ids_geodesic = jnp.concatenate([chart_id_initial,
                                              chart_ids_evolution],
                                              axis=0)

        zs_geodesic = jnp.concatenate([z_initial,
                                       zs_evolution], axis=0)

        #tuple with arrays of shape (num_steps + 1,) & (num_steps + 1, math dim)
        geodesic_states = (chart_ids_geodesic, zs_geodesic)

        return geodesic_states
