import random
import bisect
import itertools
import math
import numpy as np
from scipy import stats
import networkit
import sys
import collections
import pygirgs

from helpers.graph_analysis import shrink_to_giant_component
from helpers.powerlaw_estimation import powerlaw_fit, powerlaw_generate


# taken from https://gist.github.com/SofiaGodovykh/18f60a3b9b3e6812c071456f61f9c5a6
class UnionFind:
    """Weighted quick-union with path compression.
    The original Java implementation is introduced at
    https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf
    >>> uf = UnionFind(10)
    >>> for (p, q) in [(3, 4), (4, 9), (8, 0), (2, 3), (5, 6), (5, 9),
    ...                (7, 3), (4, 8), (6, 1)]:
    ...     uf.union(p, q)
    >>> uf._id
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    >>> uf.find(0, 1)
    True
    >>> uf._id
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    """

    def __init__(self, n):
        self._id = list(range(n))
        self._sz = [1] * n

    def _root(self, i):
        j = i
        while (j != self._id[j]):
            self._id[j] = self._id[self._id[j]]
            j = self._id[j]
        return j

    def find(self, p, q):
        return self._root(p) == self._root(q)
    
    def union(self, p, q):
        i = self._root(p)
        j = self._root(q)
        if i == j:
            return
        if (self._sz[i] < self._sz[j]):
            self._id[i] = j
            self._sz[j] += self._sz[i]
        else:
            self._id[j] = i
            self._sz[i] += self._sz[j]


# "Random ternary tree"
def random_ternary_tree(n):
    t = networkit.Graph(n)
    
    vertices = list(range(n))
    random.shuffle(vertices)

    def gen_split(start, end):
        if end-start==1:
            return vertices[start]
        if end-start==0:
            return None
        cur_root = vertices[start]
        sizes = np.random.multinomial(end-start-1, [1/3.]*3)
        n_first, n_second = sizes[0], sizes[0] + sizes[1]
        root_left = gen_split(start+1, start+1+n_first)
        root_middle = gen_split(start+1+n_first, start+1+n_second)
        root_right = gen_split(start+1+n_second, end)
        if root_left is not None:
            t.addEdge(vertices[cur_root], vertices[root_left])
        if root_middle is not None:
            t.addEdge(vertices[cur_root], vertices[root_middle])
        if root_right is not None:
            t.addEdge(vertices[cur_root], vertices[root_right])
        
        return cur_root
    
    gen_split(0, n)

    return t


# "Random binary tree"
def random_binary_tree(n):
    t = networkit.Graph(n)
    
    vertices = list(range(n))
    random.shuffle(vertices)

    def gen_split(start, end):
        if end-start==1:
            return vertices[start]
        if end-start==0:
            return None
        cur_root = vertices[start]
        n_first = np.random.binomial(end-start-1, 1/2)
        root_left = gen_split(start+1, start+1+n_first)
        root_right = gen_split(start+1+n_first, end)
        if root_left is not None:
            t.addEdge(vertices[cur_root], vertices[root_left])
        if root_right is not None:
            t.addEdge(vertices[cur_root], vertices[root_right])
        
        return cur_root
    
    gen_split(0, n)

    return t


# Connect every vertex to a random vertex in a different component
# Runtime: O(n log n)
def better_random_tree(n):
    t = networkit.Graph(n)
    nodes = t.nodes()
    vertices = list(range(n))
    random.shuffle(vertices)

    uf = UnionFind(n)

    # Connect every vertex (except for the last one) to a random vertex from another component
    for i in range(n-1):
        candidate = i
        while uf.find(i, candidate):
            candidate = random.randrange(n)
        uf.union(i, candidate)
        t.addEdge(nodes[vertices[i]], nodes[vertices[candidate]])

    return t


def random_weighted_tree(weights):
    n = len(weights)
    s = sum(weights)
    probs = [w/s for w in weights]

    t = networkit.Graph(n)
    nodes = t.nodes()

    uf = UnionFind(n)

    # Connect every vertex (except for the last one) to a random vertex from another component
    # Sorted descending by weight
    for i in sorted(range(n), key=lambda i: weights[i], reverse=True)[:-1]:
        candidate = i
        while uf.find(i, candidate):
            candidate = np.random.choice(n, p=probs)
        uf.union(i, candidate)
        t.addEdge(nodes[i], nodes[candidate])

    return t


# Generate a random tree, return the graph
# "Uniform random recursive tree"
def random_tree(n):
    t = networkit.Graph(n)
    nodes = t.nodes()
    vertices = list(range(n))
    random.shuffle(vertices)
    visited_count = 1
    while visited_count < n:
        pos1 = random.randrange(visited_count)
        pos2 = visited_count
        t.addEdge(nodes[vertices[pos1]], nodes[vertices[pos2]])
        visited_count += 1
    return t

def random_weighted(choices, weights):
    if len(choices)==1:
        return choices[0]

    cumdist = list(itertools.accumulate(weights))
    x = random.random() * cumdist[-1]
    return choices[bisect.bisect(cumdist, x)]

# Connect all components like a tree
# For each component, choose vertex uniformly at random
def make_connected_tree(g):
    comp = networkit.components.ConnectedComponents(g)
    comp.run()
    components = comp.getComponents()
    
    t = better_random_tree(len(components))
    
    for u, v in t.edges():
        g.addEdge(random.choice(components[u]), random.choice(components[v]))


# Connect all components like a tree
# Create the tree weighted by component size
# Choose random vertex for each component
def make_connected_unweighted(g):
    comp = networkit.components.ConnectedComponents(g)
    comp.run()
    components = comp.getComponents()
    
    t = random_weighted_tree(list(map(len, components)))
    
    for u, v in t.edges():
        g.addEdge(random.choice(components[u]), random.choice(components[v]))


# Connect all components like a tree
# Create the tree weighted by degree sum
# Choose random vertex each time, weighted by degree
def make_connected_weighted(g):
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    comp = networkit.components.ConnectedComponents(g)
    comp.run()
    components = comp.getComponents()
    degs_by_comp = [[degrees[i] for i in cur_comp] for cur_comp in components]
    t = random_weighted_tree([sum(degs)+len(degs) for degs in degs_by_comp])
    for u, v in t.edges():
        g.addEdge(random_weighted(components[u], degs_by_comp[u]), random_weighted(components[v], degs_by_comp[v]))
            

def binary_search(goal_f, goal, a, b, f_a=None, f_b=None, depth=0):
    if f_a is None:
        f_a = goal_f(a)
    if f_b is None:
        f_b = goal_f(b)
    m = (a + b) / 2
    f_m = goal_f(m)
    if depth < 10 and (f_a <= f_m <= f_b or f_a >= f_m >= f_b):
        if f_a <= goal <= f_m or f_a >= goal >= f_m:
            return binary_search(
                goal_f, goal,
                a, m, f_a, f_m,
                depth=depth+1)
        else:
            return binary_search(
                goal_f, goal,
                m, b, f_m, f_b,
                depth=depth+1)
    return min([(a, f_a), (b, f_b), (m, f_m)], key=lambda x: x[1])


def gradient_descent(params_measurer, params_validator, result_mergers, loss_measurers, params_updaters, generator, start, target, iterations, weights=None, samples=1):
    #if weights is None:
    #    weights = np.ones(len(start))
    #cost_limit = 0.01
    #step_size = 0.5
    #params = np.array(start, dtype=float)
    params = list(start)
    #target = np.array(target)
    #hypothesis = np.copy(target)
    hypothesis = list(target)
    for _ in range(iterations):
        # Update params based on gradient
        #gradient = hypothesis - target
        #params -= step_size * weights * gradient

        params = [params_updater(param, hypo, targ) for params_updater, param, hypo, targ in zip(params_updaters, params, hypothesis, target)]

        # Validate params, e.g., range checks
        params = list(params_validator(*params))

        # Measure current difference and cost
        results = [params_measurer(generator(*params)) for _ in range(samples)]
        results = list(map(list, zip(*results)))
        hypothesis = [merger(vals) for merger, vals in zip(result_mergers, results)]
        #hypothesis = np.mean(results, axis=0)
        #cost = np.max(np.abs((hypothesis-target) / target))
        cost = [loss_measurer(hypo, tar) for loss_measurer, hypo, tar in zip(loss_measurers, hypothesis, target)]
        print("Params: {}, result: {}, cost: {}".format(params, hypothesis, cost), file=sys.stderr)
        #if cost < cost_limit:
        #    break
    
    return params, cost


def generate_er(target_n, target_m, iterations=20, samples=10):
    def generator(n, m):
        p = 2*m/(n*(n-1))
        g = networkit.generators.ErdosRenyiGenerator(n, p).generate()
        g = shrink_to_giant_component(g)
        return g
    
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    step_size = 1.0
    loss_limit = 0.01

    n = target_n
    m = target_m

    for i in range(iterations):
        n_ = []
        m_ = []
        for _ in range(samples):
            g = generator(n, m)
            measured_n, measured_m = g.size()
            n_.append(measured_n)
            m_.append(measured_m)
        
        avg_n = sum(n_) / samples
        avg_m = sum(m_) / samples

        loss_n = abs(avg_n - target_n) / target_n
        loss_m = abs(avg_m - target_m) / target_m

        print("Loss: {} {}".format(loss_n, loss_m))

        if loss_n <= loss_limit and loss_m <= loss_limit:
            break

        if i < iterations-1:
            n += step_size * (target_n - avg_n)
            m += step_size * (target_m - avg_m)
            step_size *= 0.9

            n = max(1, n)
            m = max(1, m)

    info_map = [
        ("n", n),
        ("m", m),
        ("loss_n", loss_n),
        ("loss_m", loss_m)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])

    return info, generator(n, m)


def fit_er(g):
    n, m = g.size()
    
    return generate_er(n, m)


def generate_ba(n, m, fully_connected_start):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    m_0 = math.ceil(m / n)
    ba = networkit.Graph(n)
    nodes = ba.nodes()
    edges_added = 0
    if fully_connected_start:
        start_connections = itertools.combinations(nodes[:m_0], 2)
    else:  # circle
        start_connections = (
            [(nodes[m_0-1], nodes[0])] +
            [(nodes[i], nodes[i+1]) for i in range(m_0-1)]
        )
    for u, v in start_connections:
        ba.addEdge(u, v)
        edges_added += 1

    for i, v in list(enumerate(nodes))[m_0:]:
        num_new_edges = min(i, int((m-edges_added)/(n-i)))
        to_connect = set()
        while len(to_connect) < num_new_edges:
            num_draws = num_new_edges - len(to_connect)
            to_connect_draws = [
                random.choice(ba.randomEdge())
                for i in range(num_draws)
            ]
            to_connect |= set(
                u for u in to_connect_draws if not ba.hasEdge(v, u)
            )
        for u in to_connect:
            ba.addEdge(u, v)
        edges_added += num_new_edges
    return ba


def fit_ba(g, fully_connected_start):
    n, m = g.size()
    return generate_ba(n, m, fully_connected_start)


def generate_chung_lu(n, degrees, iterations=20, samples=5):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    initial_params = (n, degrees)
    target_params = (n, degrees)

    def generator(n, degrees):
        n = int(n)
        degree_sequence = random.choices(degrees, k=n)
        g = networkit.generators.ChungLuGenerator(degree_sequence).generate()
        g = shrink_to_giant_component(g)
        return g

    def params_validator(n, degrees):
        n = max(1, n)
        # TODO Validate degrees?
        return n, degrees

    def params_measurer(g):
        measured_n, _ = g.size()
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        return measured_n, degrees

    def degrees_merger(degree_lists):
        # Make sure all have the same length
        longest = max(map(len, degree_lists))
        for deg_list in degree_lists:
            deg_list.extend([0] * (longest - len(deg_list)))

        merged = np.round(np.mean(np.array([np.array(deg_list) for deg_list in degree_lists]), axis=0))
        return sorted(list(merged), reverse=True)

    result_mergers = [np.mean, degrees_merger]

    def standard_loss(hypothesis, target):
        return np.abs((hypothesis-target) / target)

    def degree_loss(hypothesis, target):
        statistic, p_value = stats.ks_2samp(hypothesis, target)
        return statistic

    loss_measurers = [standard_loss, degree_loss]
    step_size = 0.5

    params_updaters = [lambda n, hypo, target: n - step_size * (hypo-target), lambda degs, hypo, target: degs]


    best_params, cost = gradient_descent(result_mergers=result_mergers, loss_measurers=loss_measurers, params_updaters=params_updaters, params_measurer=params_measurer, params_validator=params_validator, generator=generator, start=initial_params, target=target_params, iterations=iterations, samples=samples)
    n, degrees = best_params

    info_map = [
        ("n", n)#,
        #("m", m)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return info, generator(*best_params)

def generate_chung_lu_constant(n, k, gamma, iterations=20, samples=5):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    initial_params = (n, k, gamma)
    target_params = (n, k, gamma)

    def generator(n, k, gamma):
        n = int(n)
        degree_sequence = powerlaw_generate(n, k, gamma)
        g = networkit.generators.ChungLuGenerator(degree_sequence).generate()
        g = shrink_to_giant_component(g)
        return g

    def params_validator(n, k, gamma):
        n = max(1, n)
        k = max(0, k)
        gamma = max(2.1, gamma)
        return n, k, gamma

    def params_measurer(g):
        n, m = g.size()
        k = (2 * m / n)
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        gamma = powerlaw_fit(degrees)
        return n, k, gamma

    best_params, cost = gradient_descent(params_measurer, params_validator, generator, initial_params, target_params, iterations=iterations, samples=samples)
    n, k, gamma = best_params

    info_map = [
        ("n", n),
        ("k", k),
        ("gamma", gamma)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return info, generator(*best_params)


def fit_chung_lu(g):
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    n, m = g.size()
    return generate_chung_lu(n, degrees)


def fit_chung_lu_constant(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)

    n, m = g.size()
    k = 2 * m / n

    return generate_chung_lu_constant(n, k, gamma)
 

def generate_hyperbolic_gd(n, m, gamma, cc, iterations=20, samples=5):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    k = (2 * m / n)
    target_params = (n, k, gamma, cc)
    initial_params = (n, k, gamma, 0.5)
    weights = np.array([1, 1, 1, -1])

    def generator(n, k, gamma, t):
        g = networkit.generators.HyperbolicGenerator(n, k, gamma, t).generate()
        g = shrink_to_giant_component(g)
        return g

    def params_validator(n, k, gamma, t):
        n = max(1, n)
        k = max(0, k)
        gamma = max(2.1, gamma)
        t = np.clip(t, 0.01, 0.99)
        return n, k, gamma, t

    def params_measurer(g):
        n, m = g.size()
        k = (2 * m / n)
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        gamma = powerlaw_fit(degrees)
        cc = networkit.globals.clustering(g)
        return n, k, gamma, cc

    best_params, cost = gradient_descent(params_measurer, params_validator, generator, initial_params, target_params, weights=weights, iterations=iterations, samples=samples)
    n, k, gamma, t = best_params
    info_map = [
        ("n", n),
        ("k", k),
        ("gamma", gamma),
        ("t", t),
        ("cost", cost)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    print("Final cost: {}".format(cost), file=sys.stderr)
    return info, generator(*best_params)


def generate_hyperbolic(n, m, gamma, cc, connected):
    def criterium(h):
        val = networkit.globals.clustering(h)
        return val


    def guess_goal(t):
        iterations = 10
        results = []
        for _ in range(iterations):
            hyper_t = networkit.generators.HyperbolicGenerator(
                n, k, gamma, t).generate()
            if connected:
                make_connected_weighted(hyper_t)
            hyper_t = shrink_to_giant_component(hyper_t)
            results.append(criterium(hyper_t))
        return sum(results)/len(results)
    

    if connected:
        fit_iterations = 2
        components = 1
        for _ in range(fit_iterations):
            m_ = m - (components - 1)
            k = 2 * m_ / n
            t, crit_diff = binary_search(guess_goal, cc, 0.01, 0.99)
            hyper = networkit.generators.HyperbolicGenerator(n, k, gamma, t).generate()
            comp = networkit.components.ConnectedComponents(hyper)
            comp.run()
            components = comp.numberOfComponents()
            
        m_ = m - (components - 1)
        k = 2 * m_ / n
        t, crit_diff = binary_search(guess_goal, cc, 0.01, 0.99)
        hyper = networkit.generators.HyperbolicGenerator(n, k, gamma, t).generate()
        print("{} components, {} out of {} remaining".format(components, m_, m))
        make_connected_weighted(hyper)
    else:
        k = 2 * m / n
        t, crit_diff = binary_search(guess_goal, cc, 0.01, 0.99)
        hyper = networkit.generators.HyperbolicGenerator(n, k, gamma, t).generate()
    
    info_map = [
        ("n", n),
        ("k", k),
        ("gamma", gamma),
        ("t", t)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, hyper)


def fit_hyperbolic(g):
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)
    n, m = g.size()
    cc = networkit.globals.clustering(g)

    return generate_hyperbolic_gd(n, m, gamma, cc)


def calc_girg(dimension, n, k, alpha, ple):
    wseed = random.randrange(10000)
    pseed = random.randrange(10000)
    sseed = random.randrange(10000)

    generator = pygirgs.Generator()
    generator.set_weights(n, -ple, wseed)
    generator.set_positions(n, dimension, pseed)
    generator.scale_weights(k, dimension, alpha)
    generator.generate(alpha, sseed)

    girg = networkit.Graph(n)
    nodes = girg.nodes()

    for u, v in generator.edge_list():
        girg.addEdge(nodes[u], nodes[v])
    
    return girg


def calc_girg_dist(dimension, degrees, k, alpha, ple):
    pseed = random.randrange(10000)
    sseed = random.randrange(10000)
    n = len(degrees)

    generator = pygirgs.Generator()
    generator.set_weights(degrees)
    generator.set_positions(n, dimension, pseed)
    generator.scale_weights(k, dimension, alpha)
    generator.generate(alpha, sseed)

    girg = networkit.Graph(n)
    nodes = girg.nodes()

    for u, v in generator.edge_list():
        girg.addEdge(nodes[u], nodes[v])
    
    return girg


def generate_girg(dimension, n, m, cc, ple, connected):

    def criterium(h):
        val = networkit.globals.clustering(h)
        return val

    def guess_goal(t):
        iterations = 10
        results = []
        for _ in range(iterations):
            girg = calc_girg(dimension, n, k, t, ple)
            if connected:
                make_connected_weighted(girg)
            girg = shrink_to_giant_component(girg)
            results.append(criterium(girg))
        return sum(results)/len(results)

    if connected:
        fit_iterations = 2
        components = 1
        for _ in range(fit_iterations):
            m_ = m - (components - 1)
            k = 2 * m_ / n
            t, crit_diff = binary_search(guess_goal, cc, 1.01, 9.0)
            girg = calc_girg(dimension, n, k, t, ple)
            comp = networkit.components.ConnectedComponents(girg)
            comp.run()
            components = comp.numberOfComponents()
            
        m_ = m - (components - 1)
        k = 2 * m_ / n
        t, crit_diff = binary_search(guess_goal, cc, 1.01, 9.0)
        girg = calc_girg(dimension, n, k, t, ple)
        print("{} components, {} out of {} remaining".format(components, m_, m))
        make_connected_weighted(girg)
    else:
        k = 2 * m / n
        t, crit_diff = binary_search(guess_goal, cc, 1.01, 9.0)
        girg = calc_girg(dimension, n, k, t, ple)
    
    info_map = [
        ("n", n),
        ("k", k),
        ("gamma", ple),
        ("t", t),
        ("dimension", dimension)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, girg)


def generate_girg_dist(dimension, degrees, cc, ple, connected):
    n = len(degrees)
    m = sum(degrees) // 2

    def criterium(h):
        val = networkit.globals.clustering(h)
        return val

    def guess_goal(t):
        iterations = 10
        results = []
        for _ in range(iterations):
            girg = calc_girg_dist(dimension, degrees, k, t, ple)
            if connected:
                make_connected_weighted(girg)
            girg = shrink_to_giant_component(girg)
            results.append(criterium(girg))
        return sum(results)/len(results)

    if connected:
        fit_iterations = 2
        components = 1
        for _ in range(fit_iterations):
            m_ = m - (components - 1)
            k = 2 * m_ / n
            t, crit_diff = binary_search(guess_goal, cc, 1.01, 9.0)
            girg = calc_girg_dist(dimension, degrees, k, t, ple)
            comp = networkit.components.ConnectedComponents(girg)
            comp.run()
            components = comp.numberOfComponents()
            
        m_ = m - (components - 1)
        k = 2 * m_ / n
        t, crit_diff = binary_search(guess_goal, cc, 1.01, 9.0)
        girg = calc_girg_dist(dimension, degrees, k, t, ple)
        print("{} components, {} out of {} remaining".format(components, m_, m))
        make_connected_weighted(girg)
    else:
        k = 2 * m / n
        t, crit_diff = binary_search(guess_goal, cc, 1.01, 9.0)
        girg = calc_girg_dist(dimension, degrees, k, t, ple)
    
    info_map = [
        ("n", n),
        ("k", k),
        ("gamma", ple),
        ("t", t),
        ("dimension", dimension)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, girg)


def fit_girg(g, dimension=1, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    degrees = networkit.centrality.DegreeCentrality(g).run().scores()

    n, m = g.size()
    cc = networkit.globals.clustering(g)

    alpha = powerlaw_fit(degrees)
    ple = max(alpha, 2.1)

    return generate_girg(dimension, n, m, cc, ple, connected)


def fit_girg_dist(g, dimension=1, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    degrees = networkit.centrality.DegreeCentrality(g).run().scores()

    cc = networkit.globals.clustering(g)

    alpha = powerlaw_fit(degrees)
    ple = max(alpha, 2.1)

    return generate_girg_dist(dimension, degrees, cc, ple, connected)
