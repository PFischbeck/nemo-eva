import random
import bisect
import itertools
import math
import numpy as np
import networkit
import collections
import pygirgs

from helpers.graph_analysis import shrink_to_giant_component
from helpers.powerlaw_estimation import powerlaw_fit


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

# Connect all other components to largest component
# Choose random vertex each time, weighted by degree
def make_connected(g):
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    comp = networkit.components.ConnectedComponents(g)
    comp.run()
    components = comp.getComponents()
    largest_comp = max(components, key=len)
    largest_comp_degs = [degrees[i] for i in largest_comp]
    for comp1 in components:
        if comp1 != largest_comp:
            comp_degs = [degrees[i] for i in comp1]
            g.addEdge(random_weighted(comp1, comp_degs), random_weighted(largest_comp, largest_comp_degs))
            

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


def generate_er(n, p, connected):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    if connected:
        p = (p*n - 2)/(n - 2)

    graph = networkit.generators.ErdosRenyiGenerator(n, p).generate()

    if connected:
        t = better_random_tree(n)
        graph.merge(t)

    return graph


def fit_er(g, connected=False):
    n, m = g.size()
    p = (2*m)/(n*(n-1))
    
    return generate_er(n, p, connected)


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

    if connected:
        tree = random_binary_tree(len(degrees))

        tree_degs = networkit.centrality.DegreeCentrality(tree).run().scores()
        tree_indices = list(range(len(degrees)))
        tree_indices.sort(key=lambda i: tree_degs[i])


        degrees.sort()
        new_degrees = [0] * len(degrees)

        diff = 0
        for tree_index, degree in zip(tree_indices, degrees):
            tree_deg = tree_degs[tree_index]

            # if degree >= tree_deg --> we can set degree accordingly and keep some of the diff
            if degree - tree_deg >= 0:
                new_degrees[tree_index] = max(0, degree - tree_deg - diff)
                diff -= min(degree - tree_deg, diff)
            else:
                new_degrees[tree_index] = 0
                diff += tree_deg - degree

        # new_degrees: degrees for chung-Lu generation, in the same order as in the tree
        cl_to_tree = list(range(len(degrees)))
        cl_to_tree.sort(key=lambda i: new_degrees[i], reverse=True)
        # cl_to_tree: indices of the nodes in tree, sorted decreasing by degree for chung-lu generation
        tree_to_cl = [0] * len(degrees)
        for cl_index, tree_index in enumerate(cl_to_tree):
            tree_to_cl[tree_index] = cl_index
        graph = networkit.generators.ChungLuGenerator(new_degrees).generate()
        for u, v in tree.edges():
            if not graph.hasEdge(tree_to_cl[u], tree_to_cl[v]):
                graph.addEdge(tree_to_cl[u], tree_to_cl[v])
        return graph


    else:
        return networkit.generators.ChungLuGenerator(degrees).generate()


def generate_chung_lu_constant(n, min_deg, max_deg, k, gamma, connected):
    generator = networkit.generators.PowerlawDegreeSequence(min_deg, max_deg, -gamma)
    
    generator.setGamma(-gamma)
    generator.run()
    generator.setMinimumFromAverageDegree(max(generator.getExpectedAverageDegree(), k))
    
    degree_sequence = generator.run().getDegreeSequence(n)
    
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
       
    graph = generate_chung_lu_constant(g.numberOfNodes(), min(degrees), max(degrees), k, gamma, connected)
    
    info_map = [
        ("n", g.numberOfNodes()),
        ("gamma", gamma),
        ("k", k)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])

    return (info, graph)
        

def generate_hyperbolic(n, m, gamma, cc, connected):
    if connected:
        # TODO Improve estimate
        estimated_components = 1
        m = m - (estimated_components - 1)

    k = 2 * m / n

    def criterium(h):
        val = networkit.globals.clustering(h)
        return val
    goal = cc

    def guess_goal(t):
        iterations = 10
        results = []
        for _ in range(iterations):
            hyper_t = networkit.generators.HyperbolicGenerator(
                n, k, gamma, t).generate()
            if connected:
                make_connected(hyper_t)
            hyper_t = shrink_to_giant_component(hyper_t)
            results.append(criterium(hyper_t))
        return sum(results)/len(results)
    t, crit_diff = binary_search(guess_goal, goal, 0.01, 0.99)
    hyper = networkit.generators.HyperbolicGenerator(
        n, k, gamma, t).generate()
    if connected:
        make_connected(hyper)
    info_map = [
        ("n", n),
        ("k", k),
        ("gamma", gamma),
        ("t", t)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, hyper)

def fit_hyperbolic(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)
    n, m = g.size()
    cc = networkit.globals.clustering(g)

    return generate_hyperbolic(n, m, gamma, cc, connected)


def generate_girg(n, dimension, k, alpha, ple, wseed, pseed, sseed):
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


def fit_girg(g, dimension=1, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)
    n, m = g.size()
    n_est = n
    k = 2 * m / n_est
    def criterium(h):
        val = networkit.globals.clustering(h)
        return val
    goal = criterium(g)

    wseed = 42
    pseed = 1234
    sseed = 12345

    def guess_goal(t):
        iterations = 10
        results = []
        for _ in range(iterations):
            girg = generate_girg(n_est, dimension, k, t, gamma, wseed, pseed, sseed)
            if connected:
                make_connected(girg)
            girg = shrink_to_giant_component(girg)
            results.append(criterium(girg))
        return sum(results)/len(results)
    t, crit_diff = binary_search(guess_goal, goal, 1.01, 9.0)

    girg = generate_girg(n_est, dimension, k, t, gamma, wseed, pseed, sseed)
    if connected:
        make_connected(girg)
    info_map = [
        ("n", n_est),
        ("k", k),
        ("gamma", gamma),
        ("t", t),
        ("dimension", dimension)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, girg)


