from experiments_util import run_random_experiments

if __name__ == "__main__":
    
    config_repetitions = {
        "repetitions": 1,
        # "no_delayed_batches": [0.1, 0.2, 0.3, 0.4],
        "no_delayed_batches": [0.4],
        # "delay_label": [10, 50, 80, 100],
        "delay_label": [0],
        # "strategies": ["gdumb", "ncm", "slda"],
        # "strategies": ["EDR","EDR-ACE", "RER", "ER_f", "ER_l", "ER_2B", "ER-ACE", "ER-ACE-Agu", "RAR"],
        # "strategies": ["ER-ACE", "EDR", "RER", "ER_l", "ER_2B", "EDR-ACE"],
        "strategies": ["ER-ACE"],
        # "datasets": ["SplitMNIST", "SplitFashionMNIST", "SplitCIFAR10", "SplitCIFAR100", "SplitMiniImagenet"],
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
        "start_delay_size": 0,
        "number_delayed_batches": 1,
        "model" : "resnet18",
        "track_flops": True,
    }
   
    run_random_experiments(config_repetitions, config)