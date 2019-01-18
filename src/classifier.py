from abstract_stage import AbstractStage
from helpers import dicts_to_df, format_feature_df
from helpers.classification import Classification
from helpers.feature_sets import get_all_feature_sets

import argparse
import collections
import functools
import multiprocessing
import pandas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def run_classifier(params, df, Y, model, network_model_mask):
    features_name, features = params
    X = df[features]
    c = Classification(
        X.loc[network_model_mask],
        Y.loc[network_model_mask],
        **model
    )
    cv_acc = c.results["cv"]["accuracy"]
    return features_name, cv_acc

def classification_experiment(df, to_compare, features_collection, cores):
    Y = df["Model"]

    model = {
        # SVM
            "model": make_pipeline(StandardScaler(), SVC()),  # kernel="rbf", cache_size=500
            "cv_grid": {
                "svc__C": [10 ** exp for exp in range(5)],
                "svc__gamma": [10 ** -exp for exp in range(5)]
            }
        # {
        #     "model": DummyClassifier,
        #     "params": {"strategy": "most_frequent"}
        # }
    }

    accuracies = pandas.DataFrame()

    pool = multiprocessing.pool.Pool(cores)

    for model1, model2 in to_compare:
        network_model_mask = (df["Model"] == model1) | (df["Model"] == model2)
        
        count = 0
        total = len(features_collection)
        classifier_function = functools.partial(run_classifier, df=df, Y=Y, model=model, network_model_mask=network_model_mask)
        for features_name, cv_acc in pool.imap(classifier_function, sorted(features_collection.items())):
            accuracies.loc[model1, features_name] = cv_acc
            count += 1
            print("{}/{} feature sets done!".format(count, total))
    else:
        pool.terminate()
        return accuracies


class Classifier(AbstractStage):
    _stage = "4-classification_results"

    def __init__(self, features, to_compare, classification_name, cores=1, **kwargs):
        super(Classifier, self).__init__()
        self.features = features
        self.cores = cores
        self.to_compare = to_compare
        self.classification_name = classification_name

    def _execute(self):
        df = dicts_to_df(self.features)
        format_feature_df(df)

        print(collections.Counter(df["Model"]))

        # TODO Option to filter for some graphs
        graphs = sorted(set(df["Graph"]))

        features_collection = get_all_feature_sets(df, graphs)
        sub_df = df.loc(axis=0)[:, graphs, :]
        accuracies = \
            classification_experiment(
                sub_df,
                self.to_compare,
                features_collection,
                self.cores)
        accuracies.to_csv(
            self._stagepath + "accuracies/" + self.classification_name + ".csv",
            header=True,
            index_label="features"
        )