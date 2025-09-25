# Look Back to Move Forward: Delay-Aware Instance Selection for Online Continual Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:  
**“Look Back to Move Forward: Delay-Aware Instance Selection for Online Continual Learning”** – Accepted at XXXX.  

---

## 📖 Overview

This repository contains the code, experiments, and resources to reproduce the results presented in our paper.  
We provide scripts for data preparation, model training, evaluation, and visualization of results.

---

## 📂 Repository Structure

```shel
.
├── experiments: folder containing the code/experiments
├── experiments/experiment_delay_ocl.py: the script that contains the configuration for running the experiments
├── data : folder containing the datasets
├── src: contains the code for the machine learning library CapyMOA
└── experiments/results_ER: folder containing all results and metric calculations
```
---

## ⚙️ Installation

```bash
git clone https://github.com/username/my-project.git (update after acceptance) 
cd CapyMOA
python3 -m venv venv
source venv/bin/activate
pip install --editable ".[dev,doc]"
```

## 📊 Data

We provide scripts to download and preprocess the dataset used in the experiments:

- **Raw data:** stored in `experiments/data.`
- **Metrics results for iterations:** stored in `experiments/results_{dataset}`.


## 🧪 Evaluation

Evaluate a trained model:

- `experiments/results_ER/ocl_plots.ipynb`: plot the graphs and calculates the metrics for OCL evaluation.
- `experiments/results_ER/ocl_plots`: all plots for test-then-train Online Windowed Accuracy.


## 🔗 Citation

If you use this code, please cite our paper:

```

@inproceedings{XXXX,
  title={Paper Title},
  author={XXX},
  booktitle={XXXX},
  year={XXX}
}
```

## 📝 License (MIT)

  