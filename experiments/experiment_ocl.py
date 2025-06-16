from capymoa.ocl.datasets import SplitMNIST, TinySplitMNIST
from capymoa.ocl.evaluation import (
    ocl_train_eval_loop, ocl_delay_train_eval_loop, OCLMetrics
    )
from capymoa.ocl.strategy import (
    ExperienceReplay, ExperienceDelayReplay
    )
from capymoa.ocl.ann import WNPerceptron
from capymoa.classifier import Finetune
from plot import plot_multiple
from typing import Dict


def run():
    results: Dict[str, OCLMetrics] = {}
    # stream = TinySplitMNIST(num_tasks=10 , shuffle_tasks=True)

    stream = SplitMNIST(num_tasks=10 , shuffle_tasks=True)
    perceptron = WNPerceptron(schema=stream.schema, hidden_size=64)
    mlp = Finetune(schema=stream.schema, model=perceptron, device="cuda")

    learner_experience_replay = ExperienceReplay(
       learner=mlp,
       buffer_size=20
    )

    results["ER"] = ocl_train_eval_loop(
        learner_experience_replay,
        stream.train_loaders(batch_size=10),
        stream.test_loaders(batch_size=10),
        continual_evaluations=10,
        progress_bar=True,
        eval_window_size=128
    )

    stream = SplitMNIST(num_tasks=10 , shuffle_tasks=True)
    perceptron = WNPerceptron(schema=stream.schema, hidden_size=64)
    mlp = Finetune(schema=stream.schema, model=perceptron, device="cuda")

    learner_experience_delay_replay = ExperienceDelayReplay(
        learner=mlp,
        buffer_size=20
    )

    results["EDR"] = ocl_delay_train_eval_loop(
        learner_experience_delay_replay,
        stream.train_loaders(batch_size=10),
        stream.test_loaders(batch_size=10),
        delay=0,
        continual_evaluations=10,
        progress_bar=True,
        eval_window_size=128
    )

    #TODO: Find Window Accuracy
    # print(f"Accuracy Final: {results.accuracy_final:.4f}")
    # print(f"Accuracy Anytime: {results.anytime_accuracy_all_avg:.4f}")
    # print(f"Accuracy Window: ")

    plots = plot_multiple(results, acc_seen=True, acc_online=True)
    plots[0].savefig("plots/results_plot.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    run()