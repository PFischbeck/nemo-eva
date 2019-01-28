import pandas

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def inflate_feature_set(feature_set_aliases):
    if "Diameter" in feature_set_aliases:
        feature_set_aliases.remove("Diameter")
        a = inflate_feature_set(feature_set_aliases + ["Effective Diameter"])
        b = inflate_feature_set(feature_set_aliases + ["Diameter Max"])
        a.update(b)
        return a

    def standardize(l):
        return tuple(sorted(l))
    inflated = {}

    distributions = [f for f_alias in feature_set_aliases for f in important_groups
                     if f.endswith(f_alias)]
    normal_features = [f_alias for f_alias in feature_set_aliases
                       if not any(f.endswith(f_alias) for f in important_groups)]
    name = ",".join(feature_set_aliases)

    inflated[name + " - mean"] = standardize(
        normal_features +
        [f + ".Location.Arithmetic Mean" for f in distributions]
    )
    if distributions:
        inflated[name + " - median"] = standardize(
            normal_features +
            [f + ".Location.Median" for f in distributions]
        )
        inflated[name + " - stats"] = standardize(
            normal_features +
            [f + ".Location.Arithmetic Mean" for f in distributions] +
            [f + ".Location.Median" for f in distributions] +
            [f + ".Dispersion.Standard Deviation" for f in distributions] +
            [f + ".Location.1st Quartile" for f in distributions] +
            [f + ".Location.3rd Quartile" for f in distributions]
        )
    if distributions == ["Centrality.ClusteringCoefficient"]:
        inflated[name + " - mean+std"] = standardize(
            normal_features +
            [
                "Centrality.ClusteringCoefficient.Location.Arithmetic Mean",
                "Centrality.ClusteringCoefficient.Dispersion.Standard Deviation"
            ]
        )

    inflated = dict([(key, ["Partition.CoreDecomposition.Dispersion.Uncorrected Standard Deviation"
                            if f == "Partition.CoreDecomposition.Dispersion.Standard Deviation"
                            else f
                            for f in group]
                      )
                     for key, group in inflated.items()])
    return inflated


def generate_inflated_feature_sets(df, aliases):
    
    feature_sets = {}
    for a in aliases:
        inflated = inflate_feature_set(a)
        all_features_present = True
        for group in inflated.values():
            for f in group:
                if f not in df.columns:
                    all_features_present = False
                    print("Warning: missing feature {}".format(f))
        if all_features_present:
            feature_sets.update(inflated)
    return feature_sets


important_groups = [
    "Centrality.Degree",
    "Centrality.ClusteringCoefficient",
    "Centrality.Betweenness",
    "Centrality.Closeness",
    "Centrality.Katz",
    "Centrality.PageRank",
    "Centrality.CoreDecomposition",
    "Partition.Communities",
    "Partition.CoreDecomposition",
]


feature_order = [
    "Nodes",
    "Edges",
    "Density",
    "Effective Diameter",
    "Diameter Max",
    "Diameter Min",
    "Partition.Communities.Properties.Size",
    "Degree Distribution.Powerlaw.Alpha",
    "Degree Distribution.Powerlaw.KS Distance"
] + [
    i + ".Location.Arithmetic Mean" for i in important_groups
] + [
    i + ".Dispersion.Standard Deviation" for i in important_groups
] + [
    i + ".Location.1st Quartile" for i in important_groups
] + [
    i + ".Location.3rd Quartile" for i in important_groups
] + [
    i + ".Location.Median" for i in important_groups
] + ["Partition.CoreDecomposition.Dispersion.Uncorrected Standard Deviation"]


def print_smallest_grouped_corr(corr, labels):
    min_grouped_correlation = 1
    groups = [labels[labels == label].index for label in set(labels)]
    for cols in [i for i in groups if len(i) > 1]:
        min_grouped_correlation = min(min_grouped_correlation, corr.loc[cols, cols].min().min())
    print("smallest grouped correlation:", min_grouped_correlation)


def cc(corr, max_corr=0.99):
    g = csr_matrix(corr >= max_corr)
    n_labels, labels = connected_components(g, directed=False)
    labels = pandas.Series(labels, index=corr.index)
    print_smallest_grouped_corr(corr, labels)
    return (n_labels, labels)


def get_uncorrelated_features(df, graphs):
    # "pearson", "spearman", "kendall"
    corr = df.loc[(slice(None), graphs, slice(None)), :].corr("spearman").abs()
    n_labels, labels = cc(corr, 0.99)
    print("found", n_labels, "groups of uncorrelated features")

    # initalize each label with a feature
    label_to_feature = dict([
        (label, labels[labels == label].index[0]) for label in set(labels)
    ])
    # replace features with higher order features
    for feature in feature_order[::-1]:
        if feature in labels:
            label_to_feature[labels[feature]] = feature
    # sort feature list
    features = sorted(
        label_to_feature.values(),
        key=lambda x: (
            feature_order.index(x)
            if x in feature_order
            else len(feature_order)
        )
    )
    return [i for i in features if i in df.columns]


def get_all_feature_sets(df, graphs):
    aliases = [
        ["Nodes", "Edges", "Centrality.CoreDecomposition"],
        ["Nodes", "Edges"],
        ["Nodes", "Edges", "ClusteringCoefficient"],
        ["Nodes", "Edges", "Degree Distribution.Powerlaw.Alpha"],
        ["Nodes", "Edges", "ClusteringCoefficient", "Degree Distribution.Powerlaw.Alpha"],
        ["Nodes", "Edges", "Degree"],
        ["Nodes", "Edges", "Degree", "ClusteringCoefficient"],
        ["Nodes", "Edges", "Diameter"],
        ["Nodes", "Edges", "ClusteringCoefficient", "Diameter"],
        ["Nodes", "Edges", "ClusteringCoefficient", "Diameter", "Degree Distribution.Powerlaw.Alpha"],
        ["ClusteringCoefficient", "Diameter", "Degree Distribution.Powerlaw.Alpha"],
        ["Nodes", "Edges", "Betweenness"],
        ["Nodes", "Edges", "Degree", "Betweenness"],
        ["Nodes", "Edges", "ClusteringCoefficient", "Betweenness"],
        ["Nodes", "Edges", "Betweenness", "Closeness", "Diameter"],
        ["Betweenness", "Closeness", "Diameter"],
        ["Nodes", "Edges", "Closeness"],
        ["Nodes", "Edges", "Centrality.CoreDecomposition"],
        ["Nodes", "Edges", "Katz"],
        ["Nodes", "Edges", "PageRank"],
        ["Nodes", "Edges", "Closeness", "ClusteringCoefficient"],
        ["Nodes", "Edges", "Centrality.CoreDecomposition", "ClusteringCoefficient"],
        ["Nodes", "Edges", "Katz", "ClusteringCoefficient"],
        ["Nodes", "Edges", "PageRank", "ClusteringCoefficient"],
        ["Nodes", "Edges", "Closeness", "Degree"],
        ["Nodes", "Edges", "Centrality.CoreDecomposition", "Degree"],
        ["Nodes", "Edges", "Katz", "Degree"],
        ["Nodes", "Edges", "PageRank", "Degree"],
        ["Nodes", "Edges", "Partition.Communities"],
        ["Nodes", "Edges", "Partition.CoreDecomposition"],
        ["Nodes", "Edges", "ClusteringCoefficient", "Partition.Communities"],
        ["Nodes", "Edges", "ClusteringCoefficient", "Partition.CoreDecomposition"],
        ["Nodes", "Edges", "Centrality.CoreDecomposition", "Partition.CoreDecomposition"],
        ["Nodes"],
        ["Edges"],
        ["Centrality.CoreDecomposition"],
        ["ClusteringCoefficient"],
        ["Degree Distribution.Powerlaw.Alpha"],
        ["Degree"],
        ["Diameter"],
        ["Betweenness"],
        ["Closeness"],
        ["Katz"],
        ["PageRank"],
        ["Partition.Communities"],
        ["Partition.CoreDecomposition"]
    ]

    feature_sets = generate_inflated_feature_sets(df, aliases)
    feature_sets["all (uncorrelated)"] = get_uncorrelated_features(df, graphs)
    return feature_sets

def get_all_feature_sets_self_check(df, graphs):
    aliases = [
        ["Nodes", "Edges"],
        ["Nodes"],
        ["Edges"],
        ["ClusteringCoefficient"],
        ["Degree Distribution.Powerlaw.Alpha"],
        ["Degree"],
    ]

    feature_sets = generate_inflated_feature_sets(df, aliases)
    return feature_sets