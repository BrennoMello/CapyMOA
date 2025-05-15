from capymoa.drift.detectors import ADWIN
from capymoa.stream.generator import AgrawalGenerator
from capymoa.stream.drift import DriftStream, AbruptDrift
from capymoa.classifier import ConceptDriftMethodClassifier
from capymoa.evaluation import prequential_cd_partially_evaluation, ConceptDriftDetectorEvaluator


def run_partially():
    print("START CD EVALUATION PARTIALLY DATA STREAM")
    stream_sea2drift = DriftStream(
        stream=[
            AgrawalGenerator(classification_function=1),
            AbruptDrift(position=2000),
            AgrawalGenerator(classification_function=2),
            AbruptDrift(position=4000),
            AgrawalGenerator(classification_function=3),
            AbruptDrift(position=6000),
            AgrawalGenerator(classification_function=4),
            AbruptDrift(position=8000),
            AgrawalGenerator(classification_function=5),
        ]
    )
    cd_classifier = ConceptDriftMethodClassifier(
        schema=stream_sea2drift.get_schema(),
        moa_drift_detector="moa.classifiers.core.driftdetection.ADWINChangeDetector",
        moa_learner="moa.classifiers.trees.HoeffdingTree",
    )
    cd_evaluator = ConceptDriftDetectorEvaluator()
    results = prequential_cd_partially_evaluation(
        stream=stream_sea2drift,
        learner=cd_classifier,
        label_probability=1,
        cd_ground_truth_list=[2000, 4000, 6000, 8000],
        cd_evaluator=cd_evaluator,
        max_instances=10000,
    )

    print(results.other_metrics())

if __name__ == "__main__":
    run_partially()
