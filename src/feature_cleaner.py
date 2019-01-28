import numpy
import pandas

from functools import reduce

from abstract_stage import AbstractStage
from helpers import dicts_to_df
from helpers import feature_sets

class FeatureCleaner(AbstractStage):
    _stage = "3-cleaned_features"

    def __init__(self, features, base_model="real-world", **kwargs):
        super(FeatureCleaner, self).__init__()
        self.features = features
        self.base_model = base_model

    def _execute(self):
        df = dicts_to_df(self.features)
        df.sort_index(axis=1, inplace=True)

        # fill information
        df.columns.name = "Feature"
        df.set_index("Graph", inplace=True)
        df['Info'] = df['Info'].fillna("no info")
        object_cols = df.columns[df.dtypes == numpy.object]
        df[object_cols] = df[object_cols].astype(str)

        # clean features we don't need
        valid_cols = [
            col for col in df.axes[1]
            if col in feature_sets.feature_order + ["Graph", "Info", "Model", "Type"]
        ]
        df_features_cleaned = df[valid_cols]

        features = df_features_cleaned.columns
        print(features.name + ":", len(features), "( unfiltered:", len(df.columns), ")")

        # clean missing models
        df_real = df_features_cleaned[df_features_cleaned["Model"] == self.base_model]
        real_graphs = set(df_real.index)
        complete_graphs = real_graphs.copy()
        for model in set(df_features_cleaned["Model"]) - set(self.base_model):
            graphs_for_model = set(df_features_cleaned[df_features_cleaned["Model"] == model].index)
            if graphs_for_model != real_graphs:
                print("missing graphs for", model, "model:", real_graphs-graphs_for_model)
                complete_graphs &= graphs_for_model
        df_real = df_real.loc[complete_graphs]
        df_cleaned = df_features_cleaned.loc[complete_graphs]
        print(df_cleaned.index.name + ":", len(df_cleaned.index), "( unfiltered:", len(df_features_cleaned.index), ")")

        # clean with filter rules
        df_real = df_cleaned[df_cleaned["Model"] == self.base_model]
        filters = {
            "CC = 0": df_real["Centrality.ClusteringCoefficient.Location.Arithmetic Mean"] == 0,
            "edges < 500": df_real["Edges"] < 500,
            "nodes < 100": df_real["Nodes"] < 100
        }
        all_filters = reduce(lambda x, y: x | y, filters.values())
        format = "{:15}{:>5}"
        sep = "-"*20
        print()
        print("Filter graphs with rules:")
        print(format.format("total", len(df_real)))
        print(sep)
        for filtername, filterdf in sorted(filters.items()):
            print(format.format(filtername, filterdf.sum()), list(df_real.index[filterdf]))
        print(sep)
        print(format.format("to filter", all_filters.sum()), list(df_real.index[all_filters]))
        print(sep)
        df_rule_filtered = df_cleaned.loc[df_real[~all_filters].index].copy()
        print(format.format("total filtered", len(df_real[~all_filters])))
        assert(len(df_real[~all_filters]) == len(df_rule_filtered) / len(set(df_rule_filtered["Model"])))

        # clean notfinite features
        def notfinite(a_df):
            return a_df.isnull() | (a_df == float("inf")) | (a_df == -float("inf"))
        valid_cols = set(df_rule_filtered.columns)
        nans = numpy.where(notfinite(df_rule_filtered))
        nan_cols = set(df_rule_filtered.columns[nans[1]])
        print("filtering cols (notfinite):")
        for col in sorted(nan_cols):
            print("    " + col)
        df_finite_filtered = df_rule_filtered[list(valid_cols - nan_cols)]
        assert not notfinite(df_finite_filtered).any().any()

        valid_cols = set(df_finite_filtered.columns)
        # normalized coefficient of variation:
        variation = df_finite_filtered.std() / df_finite_filtered.mean() / (len(df_finite_filtered)-1)**0.5
        low_variation_cols = set(variation[(variation < 0.01) | variation.isnull()].index)
        print("filtering cols (low variation):")
        for col in sorted(low_variation_cols):
            print("    " + col)
        df_final = df_finite_filtered[list(valid_cols - low_variation_cols)]

        df_final["Graph"] = df_final.index
        for a_dict in df_final.to_dict("records"):
            self._save_as_csv(a_dict)
