import argparse
import csv

from generator_er_comp import GeneratorERComp
from generator_real_world import GeneratorRealWorld
from generator_er import GeneratorER
from generator_ba_circle import GeneratorBACircle
from generator_ba_full import GeneratorBAFull
from generator_chunglu import GeneratorChungLu
from generator_chunglu_constant import GeneratorChungLuConstant
from feature_cleaner import FeatureCleaner
from classifier import Classifier

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
        
    feature_cleaner = FeatureCleaner(features, base_model="real-world", cores=cores)
    feature_cleaner.execute()
    with open(FeatureCleaner.resultspath) as input_dicts_file:
        result = list(csv.DictReader(input_dicts_file))
        
    to_compare = [("ER", "real-world"), ("BA full", "real-world"), ("BA circle", "real-world"), ("chung-lu", "real-world"), ("chung-lu constant", "real-world")]
    name = "compare_all"
    classifier = Classifier(result, to_compare=to_compare, classification_name=name, cores=cores)
    classifier.execute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=1)
    args = parser.parse_args()

    experiment = run_compare_all
    experiment(args.cores)


if __name__ == "__main__":
    main()