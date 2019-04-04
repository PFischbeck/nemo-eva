import argparse
import collections
import csv
import itertools
import math
import multiprocessing
import numpy
import random
import networkit

import os

from abstract_stage import AbstractStage
from graph_cleaner import GraphCleaner
from helpers.graph_analysis import analyze, shrink_to_giant_component
from helpers.generators import generate_chung_lu_constant, fit_chung_lu_constant

def _execute_one_graph(parameters):
    n, k, gamma = parameters

    graph_type = "parameters"

    name = "CL:n={};k={};gamma={}".format(n, k, gamma)

    print("Graph", name)

    outputs = []

    model_name = "chung-lu"

    try:
        info, model = generate_chung_lu_constant(n, k, gamma)
        output = analyze(model)
        model = shrink_to_giant_component(model)
        info2, model2 = fit_chung_lu_constant(model)
        output2 = analyze(model2)
    except Exception as e:
        print("Error:", e, "for", model_name, "of", name)
    else:
        output["Graph"] = name
        output["Type"] = graph_type
        output["Model"] = model_name+"-first"
        output["Info"] = info
        outputs.append(output)

        output2["Graph"] = name
        output2["Type"] = graph_type
        output2["Model"] = model_name+"-second"
        output2["Info"] = info2
        outputs.append(output2)

    return outputs


class GeneratorChungLuSelf(AbstractStage):
    _stage = "2-features/chung-lu-self"

    def __init__(self, graph_dicts, cores=1, **kwargs):
        super(GeneratorChungLuSelf, self).__init__()
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=1)
    args = parser.parse_args()

    parameters = []
    for n in list(range(1000, 10000, 1000))+list(range(10000, 100000, 10000)):
        for k in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            for gamma in [2.1, 2.3, 2.5, 2.7, 2.9, 3.1]:
                parameters.append((n, k, gamma))

    generator = GeneratorChungLuSelf(parameters, cores=args.cores)
    generator.execute()


if __name__ == "__main__":
    main()