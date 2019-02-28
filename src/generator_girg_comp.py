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
from helpers.generators import fit_girg, generate_girg

def _execute_one_graph(parameters):

    dimension, n, m, gamma, cc = parameters

    graph_type = "parameters"

    name = "Girg:dimension={},n={},m={},gamma={},cc={}".format(dimension, n, m, gamma, cc)

    print("Graph", name)

    outputs = []

    model_name = "girg"

    try:
        info, model = generate_girg(dimension, n, m, gamma, cc, connected=False)
        output = analyze(model)
        model = shrink_to_giant_component(model)
        info2, model2 = fit_girg(model, connected=True)
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


class GeneratorGirgComp(AbstractStage):
    _stage = "2-features/girg-comp"

    def __init__(self, parameters, cores=1, **kwargs):
        super(GeneratorGirgComp, self).__init__()
        self.parameters = parameters
        self.cores = cores
        networkit.engineering.setNumberOfThreads(1)

    def _execute(self):
        count = 0
        total = len(self.parameters)
        pool = multiprocessing.pool.Pool(self.cores)
        for results in pool.imap_unordered(_execute_one_graph, self.parameters):
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

    #with open(GraphCleaner.resultspath) as input_dicts_file:
    #    graph_dicts = list(csv.DictReader(input_dicts_file))
    parameters = []
    dimension = 1
    for n in list(range(1000, 10000, 1000))+list(range(10000, 100000, 10000))+list(range(100000, 500000, 100000)):
        for d in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            m = (n * d) // 2
            for gamma in [2.1, 2.3, 2.5, 2.7, 3.1]:
                for cc in [0.2, 0.3, 0.4]:
                    parameters.append((dimension, n, m, gamma, cc))

    generator = GeneratorGirgComp(parameters, cores=args.cores)
    generator.execute()


if __name__ == "__main__":
    main()