import random
import itertools
import math
import networkit
import collections
import pygirgs

from helpers.graph_analysis import shrink_to_giant_component
from helpers.powerlaw_estimation import powerlaw_fit

# Generate a random tree, return the graph
# Based on the Aldous-Broder algorithm, but modified for a complete graph
# TODO: Currently runs in O(n log n).
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


# Connect all other components to largest component
# Choose random vertex each time
def make_connected(g):
    comp = networkit.components.ConnectedComponents(g)
    comp.run()
    components = comp.getComponents()
    largest_comp = max(components, key=len)
    for comp1 in components:
        if comp1 != largest_comp:
            g.addEdge(random.choice(comp1), random.choice(largest_comp))


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

def fit_er(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)

    if not connected:
        return networkit.generators.ErdosRenyiGenerator.fit(g).generate()
    else:
        n, m = g.size()
        p = ((2*m)/(n-1)-2)/(n-2)

        graph = networkit.generators.ErdosRenyiGenerator(n, p).generate()

        t = random_tree(n)
        graph.merge(t)

        return graph

def fit_ba(g, fully_connected_start):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    n, m = g.size()
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

def fit_chung_lu(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    g = networkit.generators.ChungLuGenerator.fit(g).generate()
    if connected:
       make_connected(g)
    return g

def fit_chung_lu_constant(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    
    k = 2 * g.numberOfEdges() / g.numberOfNodes()
    
    generator = networkit.generators.PowerlawDegreeSequence(g)

    # Use the same gamma as the other algorithms 
    gamma = max(alpha, 2.1)
    generator.setGamma(-gamma)
    generator.run()
    generator.setMinimumFromAverageDegree(max(generator.getExpectedAverageDegree(), k))
    
    degree_sequence = generator.run().getDegreeSequence(g.numberOfNodes())
    graph = networkit.generators.ChungLuGenerator(degree_sequence).generate()
    if connected:
        make_connected(graph)
    
    info_map = [
        ("n", g.numberOfNodes()),
        ("gamma", gamma),
        ("k", k)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])

    return (info, graph)
        
def fit_hyperbolic(g, connected=False):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)
    n, m = g.size()
    degree_counts = collections.Counter(degrees)
    if connected:
        n_hyper = n
    else:
        n_hyper = n + max(0, 2*degree_counts[1] - degree_counts[2])
    
    k = 2 * m / (n_hyper-1)
    def criterium(h):
        val = networkit.globals.clustering(h)
        return val
    goal = criterium(g)

    def guess_goal(t):
        hyper_t = networkit.generators.HyperbolicGenerator(
            n_hyper, k, gamma, t).generate()
        if connected:
            make_connected(hyper_t)
        hyper_t = shrink_to_giant_component(hyper_t)
        return criterium(hyper_t)
    t, crit_diff = binary_search(guess_goal, goal, 0.01, 0.99)
    hyper = networkit.generators.HyperbolicGenerator(
        n_hyper, k, gamma, t).generate()
    if connected:
        make_connected(hyper)
    info_map = [
        ("n", n_hyper),
        ("k", k),
        ("gamma", gamma),
        ("t", t)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, hyper)

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
        girg = generate_girg(n_est, dimension, k, t, gamma, wseed, pseed, sseed)
        if connected:
            make_connected(girg)
        girg = shrink_to_giant_component(girg)
        return criterium(girg)
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


