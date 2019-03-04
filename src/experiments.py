import argparse
import csv

from generator_er_comp import GeneratorERComp
from generator_chunglu_comp import GeneratorChungLuComp
from generator_hyperbolic_comp import GeneratorHyperbolicComp
from generator_girg_comp import GeneratorGirgComp
from generator_real_world import GeneratorRealWorld
from generator_er import GeneratorER
from generator_ba_circle import GeneratorBACircle
from generator_ba_full import GeneratorBAFull
from generator_chunglu import GeneratorChungLu
from generator_chunglu_constant import GeneratorChungLuConstant
from generator_hyperbolic import GeneratorHyperbolic
from generator_girg import GeneratorGIRG
from feature_cleaner import FeatureCleaner
from classifier import Classifier

def run_girg_comp(cores):
    with open(GeneratorGirgComp.resultspath) as f:
        features = list(csv.DictReader(f))
    feature_cleaner = FeatureCleaner(features, base_model="girg", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [("girg", "girg-connected")]
    name = "girg-comp"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()


def run_hyperbolic_comp(cores):
    with open(GeneratorHyperbolicComp.resultspath) as f:
        features = list(csv.DictReader(f))
    feature_cleaner = FeatureCleaner(features, base_model="hyperbolic", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [("hyperbolic", "hyperbolic-connected")]
    name = "hyperbolic-comp"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()


def run_chunglu_comp(cores):
    with open(GeneratorChungLuComp.resultspath) as f:
        features = list(csv.DictReader(f))
    feature_cleaner = FeatureCleaner(features, base_model="chung-lu", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [("chung-lu", "chung-lu-connected")]
    name = "chung-lu-comp"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()


def run_hyper_vs_girg(cores):
    with open(GeneratorHyperbolic.resultspath) as f:
        features = list(csv.DictReader(f))
    with open(GeneratorGIRG.resultspath) as f:
        features.extend(list(csv.DictReader(f)))
    feature_cleaner = FeatureCleaner(features, base_model="hyperbolic", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [
        ("girg-1d", "hyperbolic"),
        ("girg-2d", "hyperbolic"),
        ("girg-3d", "hyperbolic")
    ]
    name = "hyper-vs-girg"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()


def run_er_comp(cores):
    with open(GeneratorERComp.resultspath) as f:
        features = list(csv.DictReader(f))
    feature_cleaner = FeatureCleaner(features, base_model="ER", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [("ER", "ER-connected")]
    name = "ER-comp"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()

def run_compare_all(cores):
    features = []

    with open(GeneratorRealWorld.resultspath) as f:
        features.extend(list(csv.DictReader(f)))
        
    with open(GeneratorER.resultspath) as f:
        features.extend(list(csv.DictReader(f)))
        
    with open(GeneratorBACircle.resultspath) as f:
        features.extend(list(csv.DictReader(f)))
        
    with open(GeneratorBAFull.resultspath) as f:
        features.extend(list(csv.DictReader(f)))

    with open(GeneratorChungLu.resultspath) as f:
        features.extend(list(csv.DictReader(f)))

    with open(GeneratorChungLuConstant.resultspath) as f:
        features.extend(list(csv.DictReader(f)))

    with open(GeneratorHyperbolic.resultspath) as f:
        features.extend(list(csv.DictReader(f)))
        
    with open(GeneratorGIRG.resultspath) as f:
        features.extend(list(csv.DictReader(f)))

    feature_cleaner = FeatureCleaner(features, base_model="real-world", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [
        ("ER", "real-world"),
        ("BA full", "real-world"),
        ("BA circle", "real-world"),
        ("chung-lu", "real-world"),
        ("chung-lu constant", "real-world"),
        ("hyperbolic", "real-world"),
        ("girg-1d", "real-world"),
        ("girg-2d", "real-world"),
        ("girg-3d", "real-world")
    ]
    name = "compare_all"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()

def main():
    experiments = {
        "compare_all": run_compare_all,
        "er_comp": run_er_comp,
        "chunglu_comp": run_chunglu_comp,
        "hyper_vs_girg": run_hyper_vs_girg,
        "hyperbolic_comp": run_hyperbolic_comp,
        "girg_comp": run_girg_comp
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--experiment', choices=experiments.keys(), default="compare_all")
    args = parser.parse_args()

    experiment = experiments[args.experiment]
    experiment(args.cores)


if __name__ == "__main__":
    main()