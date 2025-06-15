from capymoa.ocl.datasets import SplitMNIST, TinySplitMNIST
from capymoa.ocl.evaluation import (
    ocl_train_eval_loop, ocl_delay_train_eval_loop
    )
from capymoa.ocl.strategy import (
    ExperienceReplay, ExperienceDelayReplay
    )
from capymoa.ocl.ann import WNPerceptron
from capymoa.classifier import Finetune

def run():
    
    # stream = TinySplitMNIST(num_tasks=10 , shuffle_tasks=True)
    stream = SplitMNIST(num_tasks=10 , shuffle_tasks=True)
    
    perceptron = WNPerceptron(schema=stream.schema, hidden_size=64)
    mlp = Finetune(schema=stream.schema, model=perceptron)

    learner = ExperienceReplay(
       learner=mlp,
       buffer_size=200
    )

    # learner = ExperienceDelayReplay(
    #     learner=mlp,
    #     buffer_size=200
    # )

    # results = ocl_train_eval_loop(
    #     learner,
    #     stream.train_loaders(batch_size=1),
    #     stream.test_loaders(batch_size=1),
    #     continual_evaluations=10,
    #     progress_bar=True,
    # )

    results = ocl_delay_train_eval_loop(
        learner,
        stream.train_loaders(batch_size=10),
        stream.test_loaders(batch_size=10),
        delay=0,
        continual_evaluations=10,
        progress_bar=True,
    )

    #TODO: Find Window Accuracy
    print(f"Accuracy Final: {results.accuracy_final:.4f}")
    print(f"Accuracy Anytime: {results.anytime_accuracy_all_avg:.4f}")
    print(f"Accuracy Window: ")

if __name__ == "__main__":
    run()