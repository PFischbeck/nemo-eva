import random
import bisect
import itertools
import math
import numpy as np
import networkit
import sys
import statistics
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


def gradient_descent(params_measurer, generator, start, target, iterations, weights=None, samples=1):
    if not weights:
        weights = [1] * len(start)
    limit = 0.01
    alpha = 0.5
    params = start
    for _ in range(iterations):
        hypothesis = [[] for _ in range(len(params))]
        for j in range(samples):
            h = params_measurer(generator(*params))
            for i, v in enumerate(h):
                hypothesis[i].append(v)
        for i, vals in enumerate(hypothesis):
            hypothesis[i] = statistics.median(vals)
        cost = max([abs(hypo-targ) / targ for hypo, targ in zip(hypothesis, target)])
        print("Params: {}, result: {}, cost: {}".format(params, hypothesis, cost), file=sys.stderr)
        if cost < limit:
            break
        loss = [hypo-targ for hypo, targ in zip(hypothesis, target)]
        gradient = loss

        params = [param - weight * alpha * g for weight, param, g in zip(weights, params, gradient)]
    
    hypothesis = params_measurer(generator(*params))
    final_cost = max([abs(hypo-targ) / targ for hypo, targ in zip(hypothesis, target)])
    return params, final_cost


def generate_er_gd(n, m):
    params = (n, m)
    def generator(n, m):
        g = generate_er(n, m, False)
        g = shrink_to_giant_component(g)
        return g
    def params_measurer(g):
        n, m = g.size()
        return n, m
    iterations = 20
    best_params, cost = gradient_descent(params_measurer, generator, params, params, iterations)
    return generator(*best_params)


def generate_er(n, m, connected):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)


    if connected:
        fit_iterations = 2
        components = 1
        for _ in range(fit_iterations):
            m_ = m - (components - 1)
            p = (2*m_)/(n*(n-1))
            graph = networkit.generators.ErdosRenyiGenerator(n, p).generate()
            comp = networkit.components.ConnectedComponents(graph)
            comp.run()
            components = comp.numberOfComponents()
            
        m_ = m - (components - 1)
        p = (2*m_)/(n*(n-1))
        graph = networkit.generators.ErdosRenyiGenerator(n, p).generate()
        make_connected_unweighted(graph)
        print("{} components, {} out of {} remaining".format(components, m_, m))
    
    else:
        p = (2*m)/(n*(n-1))
        graph = networkit.generators.ErdosRenyiGenerator(n, p).generate()

    return graph


def fit_er(g, connected=False):
    n, m = g.size()
    
    return generate_er(n, m, connected)


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


def generate_chung_lu(degrees, connected):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    degrees.sort(reverse=True)

    if connected:
        fit_iterations = 2
        components = 1
        for _ in range(fit_iterations):
            # (Fairly) reduce degrees of high-degree vertices
            new_degrees = degrees.copy()
            diff = (components-1)*2
            while diff:
                pos = random.randrange(len(new_degrees))
                if new_degrees[pos] > 0: 
                    new_degrees[pos] -= 1
                    diff -= 1
            graph = networkit.generators.ChungLuGenerator(new_degrees).generate()
            comp = networkit.components.ConnectedComponents(graph)
            comp.run()
            components = comp.numberOfComponents()
            
        new_degrees = degrees.copy()
        diff = (components-1)*2
        while diff:
            pos = random.randrange(len(new_degrees))
            if new_degrees[pos] > 0: 
                new_degrees[pos] -= 1
                diff -= 1
        graph = networkit.generators.ChungLuGenerator(new_degrees).generate()
        make_connected_weighted(graph)
        return graph
    else:
        return networkit.generators.ChungLuGenerator(degrees).generate()


def generate_chung_lu_constant(n, max_deg, k, gamma, connected):
    degree_sequence = powerlaw_generate(n, max_deg, k, gamma)
    graph = generate_chung_lu(degree_sequence, connected)

    return graph

def fit_chung_lu(g, connected=False):
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    return generate_chung_lu(degrees, connected)


def fit_chung_lu_constant(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)
    
    k = 2 * g.numberOfEdges() / g.numberOfNodes()
       
    graph = generate_chung_lu_constant(g.numberOfNodes(), max(degrees), k, gamma, connected)
    
    info_map = [
        ("n", g.numberOfNodes()),
        ("gamma", gamma),
        ("k", k)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, graph)
 

def generate_hyperbolic_gd(n, m, gamma, cc, iterations=20, samples=5):
    k = (2 * m / n)
    target_params = (n, k, gamma, cc)
    initial_params = (n, k, gamma, 0.5)
    weights = (1, 1, 1, -1)
    def generator(n, k, gamma, t):
        g = networkit.generators.HyperbolicGenerator(n, k, gamma, t).generate()
        g = shrink_to_giant_component(g)
        return g
    def params_measurer(g):
        n, m = g.size()
        k = (2 * m / n)
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        gamma = powerlaw_fit(degrees)
        cc = networkit.globals.clustering(g)
        return n, k, gamma, cc
    best_params, cost = gradient_descent(params_measurer, generator, initial_params, target_params, weights=weights, iterations=iterations, samples=samples)
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
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
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
