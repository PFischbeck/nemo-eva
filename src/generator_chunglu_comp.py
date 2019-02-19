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
    n, min_deg, max_deg, k, gamma = parameters

    graph_type = "parameters"

    name = "CL:n={},min_deg={};max_deg={};k={};gamma={}".format(n, min_deg, max_deg, k, gamma)

    print("Graph", name)

    outputs = []

    model_name = "chung-lu"

    try:
        info, model = "", generate_chung_lu_constant(n, min_deg, max_deg, k, gamma, connected=False)
        output = analyze(model)
        model = shrink_to_giant_component(model)
        info2, model2 = fit_chung_lu_constant(model, connected=True)
        output2 = analyze(model2)
    except Exception as e:
        print("Error:", e, "for", model_name, "of", name)
    else:
        output["Graph"] = name
        output["Type"] = graph_type
        output["Model"] = model_name
        output["Info"] = info
        outputs.append(output)

        output2["Graph"] = name
        output2["Type"] = graph_type
        output2["Model"] = model_name+"-connected"
        output2["Info"] = info2
        outputs.append(output2)

    return outputs


class GeneratorChungLuComp(AbstractStage):
    _stage = "2-features/chung-lu-comp"

    def __init__(self, graph_dicts, cores=1, **kwargs):
        super(GeneratorChungLuComp, self).__init__()
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
    min_deg = 1
    for n in list(range(1000, 10000, 1000))+list(range(10000, 100000, 10000))+list(range(100000, 500000, 100000)):
        for k in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
            max_deg = int(k * 10)
            for gamma in [2.1, 2.3, 2.5, 2.7, 2.9, 3.1]:
                parameters.append((n, min_deg, max_deg, k, gamma))

    generator = GeneratorChungLuComp(parameters, cores=args.cores)
    generator.execute()


if __name__ == "__main__":
    main()