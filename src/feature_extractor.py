import collections
import csv
import itertools
import math
import multiprocessing
import numpy
import powerlaw
import random
import networkit

import os

from abstract_stage import AbstractStage
from graph_crawler import GraphCrawler
from helpers.print_blocker import PrintBlocker
from helpers.graph_analysis import shrink_to_giant_component, analyze
from helpers.generators import fit_er, fit_ba, fit_chung_lu, fit_chung_lu_constant, fit_hyperbolic, fit_girg

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
        return []

    if not g:
        print("could not import graph from path", in_path)
        return []
    if g.numberOfNodes() > 0 and g.numberOfEdges() > 0:
        if g.degree(0) == 0:
            g.removeNode(0)

    print("Graph", g.toString())
    g = shrink_to_giant_component(g)
    if g.numberOfNodes() < 100:
        print(
            "Graph is too small (" +
            str(g.numberOfNodes()) +
            " nodes, needs 100): " +
            in_path)
        return []

    model_types = [
        ("real-world",
            lambda x: ("", x)),
        ("ER",
            lambda x: ("", fit_er(x))),
        ("BA circle",
            lambda x: ("", fit_ba(x, fully_connected_start=False))),
        ("BA full",
            lambda x: ("", fit_ba(x, fully_connected_start=True))),
        ("chung-lu",
            lambda x: ("", fit_chung_lu(x))),
        ("chung-lu constant",
            fit_chung_lu_constant),
        ("hyperbolic",
            fit_hyperbolic),
        ("girg",
            fit_girg)
    ]

    outputs = []
    # all_keys = set()
    for model_name, model_converter in model_types:
        try:
            info, model = model_converter(g)
            output = analyze(model)
        except Exception as e:
            print("Error:", e, "for", model_name, "of", g.getName(), model)
        else:
            output["Graph"] = g.getName()
            output["Type"] = graph_type
            output["Model"] = model_name
            output["Info"] = info
            #self._save_as_csv(output)
            outputs.append(output)
            # all_keys |= set(output.keys())

    # for model_name, info, output in sorted(outputs):
        # for key in all_keys - set(output.keys()):
        #    output[key] = float("nan")
    return outputs


class FeatureExtractor(AbstractStage):
    _stage = "2-features"

    def __init__(self, graph_dicts, cores=1, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.graph_dicts = graph_dicts
        self.cores = cores
        networkit.engineering.setNumberOfThreads(1)

    def _execute(self):
        #for graph in self.graph_dicts:
        #    self._execute_one_graph(graph)
        count = 0
        total = len(self.graph_dicts)
        pool = multiprocessing.pool.Pool(self.cores)
        for results in pool.imap_unordered(_execute_one_graph, self.graph_dicts):
            for result in results:
                self._save_as_csv(result)
            count += 1
            print("{}/{} graphs done!".format(count, total))
        pool.close()
        pool.join()
