from capymoa.evaluation import (
    prequential_cd_delay_partially_evaluation,
    ConceptDriftDetectorEvaluator,
)
from capymoa.classifier import (
    ConceptDriftMethodClassifier, 
    HoeffdingTree,
    NaiveBayes,
    LSD,
)
from capymoa.stream.drift import (
    DriftStream, 
    AbruptDrift,
)
from capymoa.drift.detectors import (
    ADWIN
)
from capymoa.stream.generator import AgrawalGenerator

def run_cd_delay_partially():
    print("START CD EVALUATION DELAY PARTIALLY DATA STREAM")
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

    ht_classifier = HoeffdingTree(schema=stream_sea2drift.get_schema())
    naive_bayes_classifier = NaiveBayes(
        schema=stream_sea2drift.get_schema()
    )
    # sklearn_SGD = SKClassifier(
    #     schema=stream_sea2drift.get_schema(), sklearner=linear_model.SGDClassifier()
    # )
    cd_evaluator = ConceptDriftDetectorEvaluator()
    cd_classifier = ConceptDriftMethodClassifier(
        schema=stream_sea2drift.get_schema(),
        moa_drift_detector=ADWIN(),
        moa_learner=naive_bayes_classifier,
        loss=LSD,
    )
    # results = prequential_cd_delay_partially_evaluation(
    #     stream=stream_sea2drift,
    #     learner=cd_classifier,
    #     max_instances=10000,
    #     initial_window_size=0,
    #     label_probability=1,
    #     delay_probability=1,
    #     min_delay = 2,
    #     max_delay = 20,
    #     cd_ground_truth_list=[2000, 4000, 6000, 8000],
    #     cd_evaluator=cd_evaluator,
    # )

    results = prequential_cd_delay_partially_evaluation(
        stream=stream_sea2drift,
        learner=cd_classifier,
        max_instances=10000,
        initial_window_size=100,
        label_probability=1,
        delay_probability=0.5,
        min_delay = 2,
        max_delay = 20,
        cd_ground_truth_list=[2000, 4000, 6000, 8000],
        cd_evaluator=cd_evaluator,      
    )
    print("Classifier metrics results")
    print(f"learner: {results['learner']} accuracy: {results['cumulative'].accuracy()}")
    print("Drift metrics results")
    print(results.other_metrics())

if __name__ == "__main__":
    run_cd_delay_partially()
