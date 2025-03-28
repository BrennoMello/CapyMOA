from capymoa.stream.generator import RandomTreeGenerator
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.classifier import HoeffdingTree

def main():
    rtg_stream = RandomTreeGenerator()

    ht = HoeffdingTree(schema=rtg_stream.get_schema())

    results_ht = prequential_evaluation(
        max_instances=10000, window_size=1000, stream=rtg_stream, learner=ht
    )

    plot_windowed_results(results_ht, metric="accuracy")

main()