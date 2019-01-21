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
from helpers.generators import fit_chung_lu

def _execute_one_graph(graph_dict):
    in_path = (
        GraphCleaner._stagepath +
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

    print("Graph", g.toString())

    outputs = []
    model_name = "chung-lu"

    try:
        info, model = "", fit_chung_lu(g, connected=False)
        output = analyze(model)
        model = shrink_to_giant_component(model)
        info2, model2 = "", fit_chung_lu(model, connected=True)
        output2 = analyze(model2)
    except Exception as e:
        print("Error:", e, "for", model_name, "of", g.getName(), model)
    else:
        output["Graph"] = g.getName()
        output["Type"] = graph_type
        output["Model"] = model_name
        output["Info"] = info
        outputs.append(output)

        output2["Graph"] = g.getName()
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

    with open(GraphCleaner.resultspath) as input_dicts_file:
        graph_dicts = list(csv.DictReader(input_dicts_file))
    generator = GeneratorChungLuComp(graph_dicts, cores=args.cores)
    generator.execute()


if __name__ == "__main__":
    main()