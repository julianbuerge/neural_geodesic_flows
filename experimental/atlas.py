import jax
import jax.numpy as jnp


class DomainPoint(NamedTuple):
    index: int
    y_value: jnp.ndarray
    kind: str  # "interior", "boundary", or "centroid"
    neighbors: List[int] = []


class CoordinateDomain:

    def __init__(self, points: list[DomainPoint]):

        self.points = {p.index: p for p in points if p.index >= 0}

        self.centroid = next(p.y_value for p in points if p.kind == "centroid")

        self.interior_indices = [p.index for p in points if p.kind == "interior"]

        self.adjacency = {p.index: set(p.neighbors) for p in points if p.index >= 0}

        self.last_closest_idy: Optional[int] = None

    def find_closest_point_bfs(self, y: jnp.ndarray):

        if self.last_closest_idy is None:
            queue = list(self.points.keys())
        else:
            queue = [self.last_closest_idy]

        visited = set()
        best_idy = None
        best_dist = jnp.inf

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            x = self.points[current].y_value
            dist = jnp.linalg.norm(x - y)
            if dist < best_dist:
                best_dist = dist
                best_idy = current

            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)


        self.last_closest_idy = best_idy

        return best_idy

class Chart(eqx.Module):

    coordinate_domain : CoordinateDomain

    psi: callable
    phi: callable
    g: callable

    def __init__(self, psi , phi , g):

        self.psi = psi
        self.phi = phi
        self.g = g

    def connection_coeffs(self, x):

        G = lambda x : self.g(x)

        partial_g = jax.jacfwd(G)(x)
        partial_g = jnp.transpose(partial_g, axes=(2, 0, 1))

        inverse_g = jnp.linalg.inv(self.g(x))

        term1 = jnp.einsum('ki,aib->kab', inverse_g, partial_g)  # 1st term: partial_b g_ia
        term2 = jnp.einsum('ki,bai->kab', inverse_g, partial_g)  # 2nd term: partial_a g_ib
        term3 = jnp.einsum('ki,iab->kab', inverse_g, partial_g)  # 3rd term: partial_i g_ab

        Gamma = 0.5 * (term1 + term2 - term3)

        return Gamma  #shape (m,m,m)

    def geodesic_ODE_function(self, z):

        m = z.shape[0]//2

        x = z[0:m]
        v = z[m:2*m]

        Gamma = self.connection_coeffs(x) #shape (m,m,m)

        dxbydt = v

        dvbydt = -jnp.einsum('kab, a, b -> k', Gamma, v, v) # -Gamma^k_{ab} v^a v^b

        return jnp.concatenate((dxbydt,dvbydt))

    def exp_single_time_step(self, z, dt):

        k1 = self.geodesic_ODE_function(z)
        k2 = self.geodesic_ODE_function(z + 0.5*dt*k1)
        k3 = self.geodesic_ODE_function(z + 0.5*dt*k2)
        k4 = self.geodesic_ODE_function(z + dt*k3)

        z = z + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

        return z

    def exp(self, z, t, num_steps: int):

        dt = t/num_steps

        def step_function(z, _):
            z_next = self.exp_single_time_step(z, dt)
            return z_next, None

        z, _ = jax.lax.scan(step_function, z, None, length=num_steps)

        return z

    def __call__(self, y_in, t, num_steps : int):


        z_in = self.psi(y_in)

        z_out = self.exp(z_in, t, num_steps)

        y_out = self.phi(z_out)

        return y_out

#this method turns a manifold given as a collection of data points into
#a partition of clusters. It then extends the clusters to overlaping clusters.
#these are the domains. It returns these domains, with each point marked as possibly multiple of the following
#"interior, boundary, centroid"
def create_coordinate_domains(dataset, k, extension_degree):

    #returns tuple (or better different format?) of clusters, each point marked as interior, and one as the centroid
    def k_means(dataset, k):

        pass

    #increases a cluster with neighbouring points, marked as boundary
    def extend_cluster(cluster, extension_degree):

        pass


    #find tuple (or better different format?) of initial clusters
    clusters = k_means(dataset, k)

    #extend all clusters
    ...extend_cluster(cluster, extension_degree)...


    return extended_clusters


#given a tuple of coordinate domains (partition of the dataset)
#assign a psi,phi,g function to each, based on some class initializer (which has to be an eqx.Module taking two arguments: a dictionary and a key)
#and then return a tuple of Charts:
#atlas = (chart_1, chart_2, ..., chart_k)
def initialize_chart_functions(coordinatedomains: tuple[CoordinateDomain],
                               psi_initializer: callable,
                               phi_initializer: callable,
                               g_initializer: callable,
                               psi_arguments: dict,
                               phi_arguments: dict,
                               g_arguments: dict,
                               key = jax.random.PRNGKey(0)):

    psi_key, phi_key, g_key = jax.random.split(key, 3)
    psi_keys = jax.random.split(psi_key, len(domains))
    phi_keys = jax.random.split(phi_key, len(domains))
    g_keys = jax.random.split(g_key, len(domains))

    charts = []

    for i, domain in enumerate(coordinate_domains):

        psi = psi_initializer(psi_arguments, keys[i])
        phi = phi_initializer(phi_arguments, phi_keys[i])
        g = g_initializer(g_arguments, g_keys[i])

        chart = Chart(coordinate_domain=domain, psi=psi, phi=phi, g=g)

        charts.append(chart)

    atlas = tuple(charts)

    return atlas
