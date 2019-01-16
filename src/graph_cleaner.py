import collections
import csv
import itertools
import math
import multiprocessing
import numpy
import networkit

import os

from abstract_stage import AbstractStage
from graph_crawler import GraphCrawler
from helpers.print_blocker import PrintBlocker
from helpers.graph_analysis import shrink_to_giant_component

def is_graph_ok(g):
    return 100 <= g.numberOfNodes() <= 2*10**6 and 100 <= g.numberOfEdges() <= 2*10**7

def _execute_one_graph(graph_dict):
    in_path = (
        GraphCrawler()._stagepath +
        graph_dict["Group"] + "/" +
        graph_dict["Path"])

    graph_type = graph_dict["Group"]

    g = None
    try:
        g = networkit.readGraph(
            in_path,
            networkit.Format.EdgeList,
            separator=" ",
            firstNode=0,
            commentPrefix="%",
            continuous=True)
    except Exception as e:
        print(e)
        return None

    if not g:
        print("could not import graph from path", in_path)
        return None
    if g.numberOfNodes() > 0 and g.numberOfEdges() > 0:
        if g.degree(0) == 0:
            g.removeNode(0)

    originally_weighted = g.isWeighted()
    if originally_weighted:
        g = g.toUnweighted()
    g.removeSelfLoops()
    g = shrink_to_giant_component(g)
    if not is_graph_ok(g):
        print(
            "Graph does not match tolerable size range: " +
            in_path)
        return None

    graph_dict["Nodes"] = g.numberOfNodes()
    graph_dict["Edges"] = g.numberOfEdges()
    out_path = (
        GraphCleaner._stagepath +
        graph_dict["Group"] + "/" +
        graph_dict["Path"])

    directory = os.path.dirname(out_path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            pass
    networkit.writeGraph(g, out_path, networkit.Format.EdgeList, separator=" ", firstNode=0)

    return graph_dict


class GraphCleaner(AbstractStage):
    _stage = "2-cleaned-graphs"

    def __init__(self, graph_dicts, cores=1, **kwargs):
        super(GraphCleaner, self).__init__()
        self.graph_dicts = graph_dicts
        self.cores = cores
        networkit.engineering.setNumberOfThreads(1)

    def _execute(self):
        count = 0
        skipped = 0
        total = len(self.graph_dicts)
        pool = multiprocessing.pool.Pool(self.cores)
        for result in pool.imap_unordered(_execute_one_graph, self.graph_dicts):
            if result:
                self._save_as_csv(result)
            else:
                skipped += 1
            count += 1
            print("{}/{} graphs done, {} skipped!".format(count, total, skipped))
        pool.close()
        pool.join()
