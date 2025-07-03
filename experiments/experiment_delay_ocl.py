from capymoa.ocl.datasets import SplitMNIST, TinySplitMNIST
from capymoa.ocl.evaluation import (
    ocl_train_eval_loop, ocl_delay_train_eval_loop, ocl_batch_delay_train_eval_loop, OCLMetrics
    )
from capymoa.ocl.strategy import (
    ExperienceReplay, ExperienceDelayReplay
    )
from capymoa.ocl.ann import WNPerceptron
from capymoa.classifier import Finetune
from plot import plot_multiple
from typing import Dict
import numpy as np
import random
import torch
import glob
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_debug_files():
    dir_path = "debug"  # Change to your directory

    for file_path in glob.glob(os.path.join(dir_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)

def run():
    
    clean_debug_files()
    #TODO: Delay variantion
    #TODO: Delay probability 0.8
    #TODO: Schedule de tasks {1,2}, {3,4}, {5,6}, {7,8}, {9,10}
        
    #TODO: Bug with batch_size, num_tasks, and continual_evaluations
    #TODO: Bug with higth delay_probability
    config = {
        "batch_size": 128,
        "buffer_size": 20,
        "num_tasks": 10,
        "hidden_size": 64,
        "eval_window_size": 256,
        "continual_evaluations": 5,
        "delay": 0,
        "delay_probability": 0.95,
        "min_delay": 100,
        "max_delay": 110,
        "acc_seen": False
    }

    results: Dict[str, OCLMetrics] = {}
    # stream = TinySplitMNIST(num_tasks=10 , shuffle_tasks=True)

    stream = SplitMNIST(num_tasks=config["num_tasks"] , shuffle_tasks=True)
    perceptron = WNPerceptron(schema=stream.schema, hidden_size=config["hidden_size"])
    mlp = Finetune(schema=stream.schema, model=perceptron, device="cuda")

    learner_experience_replay = ExperienceReplay(
       learner=mlp,
       buffer_size=config["buffer_size"]
    )

    results["ER"] = ocl_batch_delay_train_eval_loop(
        learner_experience_replay,
        stream.train_loaders(batch_size=config["batch_size"]),
        stream.test_loaders(batch_size=config["batch_size"]),
        delay=config["delay"],
        delay_probability=config["delay_probability"],
        min_delay=config["min_delay"],
        max_delay=config["max_delay"],
        continual_evaluations=config["continual_evaluations"],
        progress_bar=True,
        eval_window_size=config["eval_window_size"]
    )

    # stream = SplitMNIST(num_tasks=config["num_tasks"] , shuffle_tasks=True)
    # perceptron = WNPerceptron(schema=stream.schema, hidden_size=config["hidden_size"])
    # mlp = Finetune(schema=stream.schema, model=perceptron, device="cuda")
    
    # # save the bug task schedule
    # with open(f"debug/task_schedule.txt", "a") as f:
    #     f.write(f"Task schedule: {stream.task_schedule}\n")

    # learner_experience_delay_replay = ExperienceDelayReplay(
    #     learner=mlp,
    #     buffer_size=config["buffer_size"],
    # )

    # #TODO: Delay in batch
    # results["EDR"] = ocl_batch_delay_train_eval_loop(
    #     learner_experience_delay_replay,
    #     stream.train_loaders(batch_size=config["batch_size"]),
    #     stream.test_loaders(batch_size=config["batch_size"]),
    #     delay=config["delay"],
    #     delay_probability=config["delay_probability"],
    #     min_delay=config["min_delay"],
    #     max_delay=config["max_delay"],
    #     continual_evaluations=config["continual_evaluations"],
    #     progress_bar=True,
    #     eval_window_size=config["eval_window_size"]
    # )

    plots = plot_multiple(results, acc_seen=config["acc_seen"], acc_online=True)
    plots[0].savefig(f'plots/results_plot_{config["acc_seen"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["buffer_size"]}_{config["delay_probability"]}_{config["min_delay"]}_{config["max_delay"]}.png', dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    set_seed(424242)
    run()