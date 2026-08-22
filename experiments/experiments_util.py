from capymoa.ann._resnet import ResNet18
from capymoa.ocl.datasets import (
    SplitMNIST, SplitFashionMNIST, SplitCIFAR10, 
    SplitCIFAR100, SplitTinyImagenet, TinySplitMNIST,
    SplitMiniImagenet
    )
from capymoa.ocl.evaluation import (
    ocl_train_eval_delayed_loop, ocl_train_eval_mixed_delayed_loop, 
    OCLMetrics, ocl_train_eval_loop
    )
from capymoa.ocl.strategy import (
    ExperienceReplay, ExperienceDelayReplay, ExperienceReplayACE,
    ExperienceReplayAsymmetricCrossEntropy, ACELoss,
    GDumb, NCM, SLDA, RAR, EWC, DER, DERPP
    )
from capymoa.ann import (
    Perceptron, ResNet18
    )

from torchvision.transforms import Compose, v2 as T

from plot import plot_multiple, ocl_plot
from capymoa.classifier import Finetune
import plotly.express as px
from typing import Dict
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch

import glob
import json
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

def run_experiment(config: dict[str, str | int | float]):
    if config["dataset"] == "TinySplitMNIST":
        stream = TinySplitMNIST(num_tasks=config["num_tasks"], shuffle_tasks=True)
    if config["dataset"] == "SplitMNIST":
        stream = SplitMNIST(num_tasks=config["num_tasks"], shuffle_tasks=True)
    if config["dataset"] == "SplitFashionMNIST":
        stream = SplitFashionMNIST(num_tasks=config["num_tasks"], shuffle_tasks=True)
    if config["dataset"] == "SplitCIFAR10":
        stream = SplitCIFAR10(num_tasks=config["num_tasks"], shuffle_tasks=True)
    if config["dataset"] == "SplitCIFAR100":
        stream = SplitCIFAR100(num_tasks=config["num_tasks"], shuffle_tasks=True)
    if config["dataset"] == "SplitTinyImagenet":
        stream = SplitTinyImagenet(num_tasks=config["num_tasks"], shuffle_tasks=True)
    if config["dataset"] == "SplitMiniImagenet":
        stream = SplitMiniImagenet(num_tasks=config["num_tasks"], shuffle_tasks=True)
   
    log_task_schedule(stream.task_schedule)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["model"] == "resnet18":
        if config["dataset"] == "SplitMNIST" or config["dataset"] == "SplitFashionMNIST":
            ann = ResNet18(num_classes=stream.schema.get_num_classes(), 
                                in_channels=1, weights=None, small_input=True)
        elif config["dataset"] == "SplitTinyImagenet" or config["dataset"] == "SplitMiniImagenet":
            ann = ResNet18(num_classes=stream.schema.get_num_classes(), 
                                in_channels=3, weights=None, small_input=False)
        else:
            ann = ResNet18(num_classes=stream.schema.get_num_classes(), 
                                in_channels=3, weights=None, small_input=True)
    else:        
        ann = Perceptron(schema=stream.schema, hidden_size=config["hidden_size"])


    finetune = Finetune(schema=stream.schema, model=ann, device=device)

    if config["strategy"] in ["RER", "ER_f", "ER_l", "ER_2B"]:
        learner_experience = ExperienceReplay(
            learner=finetune,
            buffer_size=config["buffer_size"]
        )
    elif config["strategy"] == "EDR":
        learner_experience = ExperienceDelayReplay(
            learner=finetune,
            buffer_size=config["buffer_size"],
        )
    elif config["strategy"] == "EDR-ACE":
        finetune_ace = Finetune(schema=stream.schema, 
                                model=ann, criterion=ACELoss(device=device), 
                                device=device)
        learner_experience = ExperienceDelayReplay(
            learner=finetune_ace,
            buffer_size=config["buffer_size"],
        )
    elif config["strategy"] == "ER-ACE-Agu":
        learner_experience = ExperienceReplayACE(
            learner=finetune,
            device=device,
            buffer_size=config["buffer_size"],
        )
    elif config["strategy"] == "ER-ACE":
        learner_experience = ExperienceReplayACE(
            learner=finetune,
            device=device,
            use_augs=False,
            buffer_size=config["buffer_size"],
        )
    elif config["strategy"] == "RAR":
        augment = T.Compose([
            T.RandomRotation(10),
        ])
        learner_experience = RAR(
                                learner=finetune,
                                coreset_size=config["buffer_size"],
                                augment=augment,
                            )
    elif config["strategy"] == "DER":
        augment = get_augumentations(config["dataset"])
        learner_experience = DER(
                                learner=finetune,
                                coreset_size=config["buffer_size"],
                                augment=augment,
                            )
        
    elif config["strategy"] == "DERPP":
        augment = get_augumentations(config["dataset"])
        learner_experience = DERPP(
                                    learner=finetune,
                                    coreset_size=config["buffer_size"],
                                    augment=augment,
                                )

    elif config["strategy"] == "gdumb":
        learner_experience = GDumb(
            schema=stream.schema,
            model=finetune,
            epochs=1,
            batch_size=config["batch_size"],
            capacity=config["buffer_size"],
            device=device,
            seed=config["seed"]
        )
    elif config["strategy"] == "ncm":
        learner_experience = NCM(
            schema=stream.schema
        )
    elif config["strategy"] == "slda":
        learner_experience = SLDA(
            schema=stream.schema
        )
    elif config["strategy"] == "EWC":
        optimiser = torch.optim.SGD(ann.parameters(), lr=0.06)
        learner_experience = EWC(
            schema=stream.schema,
            model=ann,
            optimiser=optimiser,
            device=device,
            lambda_= 0.4,  # Regularization strength
        )
    else:
        raise ValueError(f"Strategy {config['strategy']} not recognized.")
    
    if config["delay_label"] != 0:
        return ocl_train_eval_mixed_delayed_loop(
            learner_experience,
            stream.train_loaders(batch_size=config["batch_size"]),
            stream.test_loaders(batch_size=config["batch_size"]),
            continual_evaluations=config["continual_evaluations"],
            progress_bar=True,  
            eval_window_size=config["eval_window_size"],
            delayed_batches=config["delay_label"],
            select_tasks=config["select_tasks"],
            number_delayed_batches=config["number_delayed_batches"],
            prob_no_delay_batches=config["prob_no_delay_batches"],
            er_strategy=config["strategy"],
            track_flops=config["track_flops"],
        )  
    else:
        return ocl_train_eval_loop(
            learner_experience,
            stream.train_loaders(batch_size=config["batch_size"]),
            stream.test_loaders(batch_size=config["batch_size"]),
            continual_evaluations=config["continual_evaluations"],
            progress_bar=True,  
            eval_window_size=config["eval_window_size"],
        ) 

def get_augumentations(dataset: str):
    if dataset in ["SplitMNIST", "SplitFashionMNIST"]:
        augment = T.Compose([
                                nn.Identity(),
                            ])
    elif dataset == "SplitCIFAR10":
        augment = T.Compose([
                        T.RandomCrop(size=32, padding=4),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToImage(),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                    ])
    elif dataset == "SplitCIFAR100":
        augment = T.Compose([
                        T.RandomCrop(size=32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
                    ])
    elif dataset in ["SplitTinyImagenet", "SplitMiniImagenet"]:
        augment = T.Compose([
                            T.RandomCrop(size=64, padding=4),
                            T.RandomHorizontalFlip(),
                            T.ToImage(),
                            T.ToDtype(torch.float32, scale=True),
                            T.Normalize(mean=[0.4802, 0.4480, 0.3975], std=[0.2770, 0.2691, 0.2821]),
                        ])
    else:
        raise ValueError(f"Dataset {dataset} not recognized for augmentation.")
    
    return augment

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


def _save_json_results(
    dataset, delay_label, prob_no_delay_batches, batch_size, num_tasks, strategy, hidden_size, eval_window_size,
    continual_evaluations, number_delayed_batches, results_repetition, repetition
):
    os.makedirs(f"results_{dataset}", exist_ok=True)
    
    #TODO: Add anytime accuracy seen
    data_to_save = {
        "task_index": results_repetition.task_index,
        "accuracy_final": results_repetition.accuracy_final,
        "accuracy_all": results_repetition.accuracy_all,
        "accuracy_all_avg": results_repetition.accuracy_all_avg,
        "anytime_task_index": results_repetition.anytime_task_index,
        "anytime_accuracy_all": results_repetition.anytime_accuracy_all,
        "anytime_accuracy_all_avg": results_repetition.anytime_accuracy_all_avg,
        "accuracy_seen": results_repetition.accuracy_seen,
        "accuracy_seen_avg": results_repetition.accuracy_seen_avg,
        "ttt_windowed_task_index": results_repetition.ttt_windowed_task_index,
        "ttt_windowed_accuracy": results_repetition.ttt.windowed.accuracy(),
        "ttt_cumulative_accuracy": results_repetition.ttt.cumulative.accuracy(),
        "ttt_metrics_per_window_accuracy": results_repetition.ttt.metrics_per_window()['accuracy'],
        "forgetting": results_repetition.forgetting,
        "backward_transfer": results_repetition.backward_transfer,
        "forward_transfer": results_repetition.forward_transfer,
        "total_flops": results_repetition.total_flops,
    }

    for k, v in data_to_save.items():
        if isinstance(v, np.ndarray):
            data_to_save[k] = v.tolist()
        elif isinstance(v, pd.Series):
            data_to_save[k] = v.tolist()

    filename = f"results_{dataset}/results_{dataset}_{delay_label}_{prob_no_delay_batches}_{batch_size}_{num_tasks}_{strategy}_{hidden_size}_{eval_window_size}_{continual_evaluations}_{number_delayed_batches}_{repetition}.json"
    
    with open(filename, "w") as f:
        json.dump(data_to_save, f, indent=4)


def run_random_no_delayed_experiments():
    #TODO: ER with additional loss based of ER-ACE
    config_repetitions = {
        "repetitions": 30,
        # "strategies": ["gdumb", "ncm", "slda"],
        # "strategies": ["EDR", "RER", "ER_f", "ER_l", "ER_2B", "ER-ACE", "ER-ACE-Agu"],
        "strategies": ["ER-ACE-Agu", "ER-ACE"],
        # "datasets": ["SplitMNIST", "SplitFashionMNIST", "SplitCIFAR10", "SplitCIFAR100", "SplitTinyImagenet"],
        "datasets": ["SplitCIFAR10"],
    }
    
    config = {
        "batch_size": 10,
        "buffer_size": 100,
        "num_tasks": 5,
        "hidden_size": 64,
        "eval_window_size": 128,
        "continual_evaluations": 5,
        "acc_seen": False,
        "select_tasks": [],
        "no_delayed_tasks": [],
        "delay_label": 0,
        "prob_no_delay_batches": 0.0,  
        "start_delay_size": 0,
        "number_delayed_batches": 1,
        "model" : "resnet18",
    }
    
    for dataset in config_repetitions["datasets"]:       
        config["dataset"] = dataset
        if dataset == "SplitCIFAR100":
            config["num_tasks"] = 20
        if dataset == "SplitTinyImagenet":
            config["num_tasks"] = 20
        
        for strategy in config_repetitions["strategies"]:
            config["strategy"] = strategy
            
            for repetition in range(config_repetitions["repetitions"]):
                set_seed(424242+repetition)
                config["seed"] = 424242+repetition
                
                print(f'Running {dataset} - {strategy} - repetition {repetition+1}/{config_repetitions["repetitions"]}')
                results_repetition = run_experiment(config)
                _save_json_results(
                    config["dataset"],  config["delay_label"], config["prob_no_delay_batches"], config["batch_size"], 
                    config["num_tasks"], config["strategy"], config["hidden_size"], config["eval_window_size"], 
                    config["continual_evaluations"], config["number_delayed_batches"], results_repetition, repetition
                    )
                plot_task_results(results_repetition, config)


def run_experiments():
    config_repetitions = {
        # "datasets": ["SplitCIFAR100"],
        # "strategies": ["EDR", "ER_f", "ER_2B"],
        "datasets": ["SplitMNIST", "SplitCIFAR10", "SplitCIFAR100"],
        # "strategies": ["RER", "ER_f", "ER_l", "ER_2B", "ER-ACE"],      
        # "datasets": ["SplitMNIST"],
        # "datasets": ["SplitTinyImagenet"],
        "strategies": ["ER-ACE"],
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
        "number_delayed_batches": 1,
        "prob_no_delay_batches": 0.4,
    }

    results: Dict[str, OCLMetrics] = {}
    for dataset in config_repetitions["datasets"]:
        clean_debug_files()
        config["dataset"] = dataset
        if dataset == "SplitCIFAR100":
            config["num_tasks"] = 5
            config["delay_label"] = 50

        for strategy in config_repetitions["strategies"]:
            set_seed(424242)
            
            config["strategy"] = strategy
            results[strategy] = run_experiment(config)

            plot_task_results(results[strategy], config)
            #plot_task_heatmaps(config)

        plot_online_accuracy(results, config)

def run_random_experiments(config_repetitions, config):
    
    for dataset in config_repetitions["datasets"]:       
        config["dataset"] = dataset
        if (dataset == "SplitCIFAR100" or dataset == "SplitMiniImagenet" 
            or dataset == "SplitTinyImagenet"):
            config["num_tasks"] = 20
        
        for delay in config_repetitions["delay_label"]:
            config["delay_label"] = delay
            
            for no_delayed in config_repetitions["no_delayed_batches"]:
                config["prob_no_delay_batches"] = no_delayed

                for strategy in config_repetitions["strategies"]:
                    config["strategy"] = strategy
                    
                    for repetition in range(config_repetitions["repetitions"]):
                        set_seed(424242+repetition)
                        config["seed"] = 424242+repetition
                        
                        print(f'Running {dataset} - {strategy} - repetition {repetition+1}/{config_repetitions["repetitions"]}')
                        results_repetition = run_experiment(config)
                        _save_json_results(
                            config["dataset"],  config["delay_label"], config["prob_no_delay_batches"], config["batch_size"], 
                            config["num_tasks"], config["strategy"], config["hidden_size"], config["eval_window_size"], 
                            config["continual_evaluations"], config["number_delayed_batches"], results_repetition, repetition
                            )
                        plot_task_results(results_repetition, config)