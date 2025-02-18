import random
import itertools
import math
import networkit
import collections

from helpers.graph_analysis import shrink_to_giant_component
from helpers.powerlaw_estimation import powerlaw_fit

# Generate a random tree, return the edge list
# Based on the Aldous-Broder algorithm, but modified for a complete graph
# TODO: Currently runs in O(n log n).
def random_tree(n):
    edges = []
    vertices = list(range(n))
    random.shuffle(vertices)
    current_pos = 0
    visited_count = 1
    while visited_count < n:

        # Repeat random walk until we see new vertex
        while True:
            potential_next = random.randrange(n)
            if potential_next == current_pos:
                continue
            if potential_next >= visited_count:
                break
            
            current_pos = potential_next

        next_pos = visited_count
        edges.append((vertices[current_pos], vertices[next_pos]))
        visited_count += 1
        current_pos = next_pos
    return edges


def make_connected(g):
    comp = networkit.components.ConnectedComponents(g)
    comp.run()
    components = comp.getComponents()
    for comp1, comp2 in zip(components, components[1:]):
        g.addEdge(random.choice(comp1), random.choice(comp2))


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

def fit_er(g):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    n, m = g.size()

    p = ((2*m)/(n-1)-2)/(n-2)

    graph = networkit.generators.ErdosRenyiGenerator(n, p).generate()

    tree_edges = random_tree(n)
    for u, v in tree_edges:
        if not graph.hasEdge(graph.nodes()[u], graph.nodes()[v]):
            graph.addEdge(graph.nodes()[u], graph.nodes()[v])

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

def fit_chung_lu(g):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    g = networkit.generators.ChungLuGenerator.fit(g).generate()
    make_connected(g)
    return g

def fit_chung_lu_constant(g):
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
    make_connected(graph)
    
    info_map = [
        ("n", g.numberOfNodes()),
        ("gamma", gamma),
        ("k", k)
    ]
    
    info = "|".join([name + "=" + str(val) for name, val in info_map])

    return (info, graph)
        
def fit_hyperbolic(g):
    random.seed(42, version=2)
    networkit.setSeed(seed=42, useThreadId=False)
    degrees = networkit.centrality.DegreeCentrality(g).run().scores()
    alpha = powerlaw_fit(degrees)
    gamma = max(alpha, 2.1)
    n, m = g.size()
    degree_counts = collections.Counter(degrees)
    n_hyper = n + max(0, 2*degree_counts[1] - degree_counts[2])
    k = 2 * m / (n_hyper-1)
    def criterium(h):
        #networkit.setLogLevel("WARN")
        val = networkit.globals.clustering(h)
        #networkit.setLogLevel("INFO")
        return val
    goal = criterium(g)

    def guess_goal(t):
        hyper_t = networkit.generators.HyperbolicGenerator(
            n_hyper, k, gamma, t).generate()
        make_connected(hyper_t)
        hyper_t = shrink_to_giant_component(hyper_t)
        return criterium(hyper_t)
    t, crit_diff = binary_search(guess_goal, goal, 0.01, 0.99)
    hyper = networkit.generators.HyperbolicGenerator(
        n_hyper, k, gamma, t).generate()
    make_connected(hyper)
    info_map = [
        ("n", n_hyper),
        ("k", k),
        ("gamma", gamma),
        ("t", t)
    ]
    info = "|".join([name + "=" + str(val) for name, val in info_map])
    return (info, hyper)
