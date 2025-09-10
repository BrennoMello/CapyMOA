from capymoa.ocl.datasets import SplitFashionMNIST, SplitCIFAR10, SplitCIFAR100
from capymoa.ocl.evaluation import (
    ocl_train_eval_delayed_loop, OCLMetrics
    )
from capymoa.ocl.strategy import (
    ExperienceReplay, ExperienceDelayReplay
    )
from capymoa.ann import Perceptron
from capymoa.classifier import Finetune
from plot import plot_multiple, ocl_plot
import plotly.express as px
from typing import Dict
import pandas as pd
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_task_schedule(task_schedule: Dict[int, int]):
    os.makedirs("debug", exist_ok=True)
    with open("debug/task_schedule.log", "a") as f:
        f.write(f"Task schedule: {task_schedule}\n")

def clean_debug_files():
    dir_path = "debug"  # Change to your directory

    for file_path in glob.glob(os.path.join(dir_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)

def run_experiment(config):
    if config["dataset"] == "SplitCIFAR10":
        stream = SplitCIFAR10(num_tasks=config["num_tasks"] , shuffle_tasks=True)
    if config["dataset"] == "SplitCIFAR100":
        stream = SplitCIFAR100(num_tasks=config["num_tasks"] , shuffle_tasks=True)
    if config["dataset"] == "SplitFashionMNIST":
        stream = SplitFashionMNIST(num_tasks=config["num_tasks"] , shuffle_tasks=True)
    
    log_task_schedule(stream.task_schedule)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    perceptron = Perceptron(schema=stream.schema, hidden_size=config["hidden_size"])
    mlp = Finetune(schema=stream.schema, model=perceptron, device=device)

    if config["strategy"] == "ER":
        learner_experience = ExperienceReplay(
            learner=mlp,
            buffer_size=config["buffer_size"]
        )
    if config["strategy"] == "EDR":
        learner_experience = ExperienceDelayReplay(
            learner=mlp,
            buffer_size=config["buffer_size"],
        )

    return ocl_train_eval_delayed_loop(
        learner_experience,
        stream.train_loaders(batch_size=config["batch_size"]),
        stream.test_loaders(batch_size=config["batch_size"]),
        continual_evaluations=config["continual_evaluations"],
        progress_bar=True,
        eval_window_size=config["eval_window_size"],
        delay_label=config["delay_label"],
        select_tasks=config["select_tasks"],
        no_delayed_tasks=config["no_delayed_tasks"],
        start_delay_size=config["start_delay_size"],
        number_delayed_batches=config["number_delayed_batches"],
    )

def plot_task_results(results, config):
    plots = ocl_plot(
        results,
        task_acc=True,
        online_acc=False,
        acc_all=False,
        acc_seen=config["acc_seen"],
    )
    #create folder if not exists
    folder_name = f'batch_size_{config["batch_size"]}_buffer_size_{config["buffer_size"]}_num_tasks_{config["num_tasks"]}_hidden_size_{config["hidden_size"]}_eval_window_size_{config["eval_window_size"]}_continual_evaluations_{config["continual_evaluations"]}_delay_label_{config["delay_label"]}_start_delay_size_{config["start_delay_size"]}_number_delayed_batches_{config["number_delayed_batches"]}'
    os.makedirs(f"plots/{folder_name}", exist_ok=True)

    plots[0].savefig(
        f'plots/{folder_name}/results_plot_tasks_{config["dataset"]}_{config["strategy"]}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.png',
        dpi=300, bbox_inches="tight"
    )

def plot_online_accuracy(results, config):
    plots = plot_multiple(
        results, 
        acc_seen=config["acc_seen"], 
        acc_online=True
    )
    #create folder if not exists
    folder_name = f'batch_size_{config["batch_size"]}_buffer_size_{config["buffer_size"]}_num_tasks_{config["num_tasks"]}_hidden_size_{config["hidden_size"]}_eval_window_size_{config["eval_window_size"]}_continual_evaluations_{config["continual_evaluations"]}_delay_label_{config["delay_label"]}_start_delay_size_{config["start_delay_size"]}_number_delayed_batches_{config["number_delayed_batches"]}'
    os.makedirs(f"plots/{folder_name}", exist_ok=True)

    plots[0].savefig(f'plots/{folder_name}/results_plot_online_acc_{config["dataset"]}_{["EDR", "ER"]}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.png',
                     dpi=300, bbox_inches="tight")

#TODO: plot heatmap of task and batches correlation
def plot_task_heatmaps(config):
    if config["strategy"] == "EDR":
        df = pd.read_csv("debug/train_batches_y_ExperienceDelayReplay.log", 
                         names=["task_id", "class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", 
                                "class_7", "class_8", "class_9"])
        df = df.drop(columns=["task_id"])
        heatmap = px.imshow(df.corr(), text_auto=True, title=f'Correlation Heatmap - {config["dataset"]} - {config["strategy"]} - Delay Label: {config["delay_label"]}')
        #create folder if not exists
        folder_name = f'batch_size_{config["batch_size"]}_buffer_size_{config["buffer_size"]}_num_tasks_{config["num_tasks"]}_hidden_size_{config["hidden_size"]}_eval_window_size_{config["eval_window_size"]}_continual_evaluations_{config["continual_evaluations"]}_delay_label_{config["delay_label"]}_start_delay_size_{config["start_delay_size"]}_number_delayed_batches_{config["number_delayed_batches"]}'
        os.makedirs(f"plots/{folder_name}", exist_ok=True)

        heatmap.write_image(f'plots/{folder_name}/heatmap_{config["dataset"]}_{config["strategy"]}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.png', 
                            scale=3)
        
    # df = pd.read_csv("debug/train_batches_y_ExperienceDelayReplay.log", 
    #              names=["task_id", "class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", 
    #                         "class_7", "class_8", "class_9"])

def run_experiments():
    config_repetitions = {
        "datasets": ["SplitFashionMNIST", "SplitCIFAR10", "SplitCIFAR100"],
        "strategies": ["EDR", "ER"]
        # "datasets": ["SplitFashionMNIST"]
    }
    
    config = {
        "batch_size": 32,
        "buffer_size": 128,
        "num_tasks": 5,
        "hidden_size": 64,
        "eval_window_size": 128,
        "continual_evaluations": 5,
        "delay_label": 100,
        "acc_seen": False,
        "select_tasks": [],
        "no_delayed_tasks": [],  
        "start_delay_size": 0,
        "number_delayed_batches": 2,
    }

    results: Dict[str, OCLMetrics] = {}
    for dataset in config_repetitions["datasets"]:
        clean_debug_files()
        config["dataset"] = dataset
        if dataset == "SplitCIFAR100":
            config["num_tasks"] = 10
            config["delay_label"] = 25

        for strategy in config_repetitions["strategies"]:
            set_seed(424242)
            
            config["strategy"] = strategy
            results[strategy] = run_experiment(config)

            # plots = ocl_plot(results[strategy], task_acc=True, online_acc=False, acc_all=False, acc_seen=False, )
            # plots[0].savefig(f'plots/results_plot_{dataset}_{config["strategy"]}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.png', 
            #             dpi=300, bbox_inches="tight")
            plot_task_results(results[strategy], config)
            #plot_task_heatmaps(config)

        plot_online_accuracy(results, config)
        # plots = plot_multiple(results, acc_seen=config["acc_seen"], acc_online=True)
        # plots[0].savefig(f'plots/results_plot_{dataset}_{config_repetitions["strategies"]}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.png', 
        #                 dpi=300, bbox_inches="tight")

def run_random_experiments():
    
    config_repetitions = {
        "repetitions": 5,
        "datasets": ["SplitFashionMNIST", "SplitCIFAR10", "SplitCIFAR100"],
        "strategies": ["EDR", "ER"]
        # "datasets": ["SplitFashionMNIST"]
    }
    
    config = {
        "batch_size": 32,
        "buffer_size": 128,
        "num_tasks": 5,
        "hidden_size": 64,
        "eval_window_size": 128,
        "continual_evaluations": 5,
        "delay_label": 100,
        "acc_seen": False,
        "select_tasks": [],
        "no_delayed_tasks": [],  
        "start_delay_size": 0,
        "number_delayed_batches": 2,
    }

    accuracy_results_ER = []
    accuracy_results_EDR = []
    results: Dict[str, OCLMetrics] = {}
    for dataset in config_repetitions["datasets"]:
       
        config["dataset"] = dataset
        if dataset == "SplitCIFAR100":
            config["num_tasks"] = 10
            config["delay_label"] = 25

        for strategy in config_repetitions["strategies"]:
            config["strategy"] = strategy
            for repetition in range(config_repetitions["repetitions"]):
                set_seed(424242+repetition)
                print(f'Running {dataset} - {strategy} - repetition {repetition+1}/{config_repetitions["repetitions"]}')
                results[strategy] = run_experiment(config)

                if strategy == "ER":
                    accuracy_results_ER.append(results[strategy].ttt.cumulative.accuracy() / 100)
                if strategy == "EDR":
                    accuracy_results_EDR.append(results[strategy].ttt.cumulative.accuracy() / 100)

            #log mean and std of accuracy_results
            if strategy == "ER":
                mean_accuracy = round(np.mean(accuracy_results_ER, axis=0), 2)
                std_accuracy = round(np.std(accuracy_results_ER, axis=0), 2)
                print(f'Mean accuracy {mean_accuracy}')
                print(f'Std accuracy {std_accuracy}')
  
                np.savetxt(f'results_mean_std_{dataset}_{strategy}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.csv', 
                             np.column_stack((mean_accuracy, std_accuracy)), delimiter=",", header="mean_accuracy,std_accuracy", comments="", fmt="%.2f")
                accuracy_results_ER = []
            if strategy == "EDR":
                mean_accuracy = round(np.mean(accuracy_results_EDR, axis=0), 2)
                std_accuracy = round(np.std(accuracy_results_EDR, axis=0), 2)
                print(f'Mean accuracy {mean_accuracy}')
                print(f'Std accuracy {std_accuracy}')
  
                np.savetxt(f'results_mean_std_{dataset}_{strategy}_{config["acc_seen"]}_{config["no_delayed_tasks"]}_{config["batch_size"]}_{config["num_tasks"]}_{config["start_delay_size"]}_{config["delay_label"]}_{config["number_delayed_batches"]}.csv', 
                             np.column_stack((mean_accuracy, std_accuracy)), delimiter=",", header="mean_accuracy,std_accuracy", comments="", fmt="%.2f")
                accuracy_results_EDR = []

if __name__ == "__main__":
    run_random_experiments()