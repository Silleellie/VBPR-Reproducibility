# VBPR-replicability
Repository which includes all things related to replicate VBPR paper by Prof. Julian McAuley of 2016

## How to Use
Simply perform `pip install requirements.txt` on a freshly created *virtual environment* and then run via *command line*:

```
python pipeline.py
```

In this way, raw data will first be *downloaded* and *processed*, and then the actual experiment will be run using the ***default parameters***.
* By default, the experiment is run for $5$, $10$, $20$, $50$ ***epochs***. Default parameters can be easily changed by passing them as *command line arguments*

You can inspect all the parameters that can be set by simply running `python pipeline.py –h`. This is what you would obtain:

```
$ python pipeline.py –h

usage: pipeline.py [-h] [-epo 5 [5 ...]] [-bs 128] [-gd 20] [-td 20] [-lr 0.005] [-seed 42]

Main script to reproduce the VBPR experiment

optional arguments:
  -h, --help            show this help message and exit
  -epo 5 [5 ...], --epochs 5 [5 ...]
                        Number of epochs for which the VBPR network will be trained
  -bs 128, --batch_size 128
                        Batch size that will be used for the torch dataloaders during training
  -gd 20, --gamma_dim 20
                        Dimension of the gamma parameter of the VBPR network
  -td 20, --theta_dim 20
                        Dimension of the theta parameter of the VBPR network
  -lr 0.005, --learning_rate 0.005
                        Learning rate for the VBPR network
  -seed 42, --random_seed 42
                        random seed
```

Project Organization
------------
    ├── 📁 clayrs_can_see              <- Package containing a modified version of clayrs with VBPR support
    │
    ├── 📁 data                        <- Directory containing all data generated/used by the experiment
    │   ├── 📁 interim                      <- Intermediate data that has been transformed
    │   ├── 📁 processed                    <- The final, canonical data sets used for training
    │   └── 📁 raw                          <- The original, immutable data dump
    │
    ├── 📁 models                       <- Trained and serialized models at different epochs
    │   ├── 📁 vbpr_clayrs                  <- Models which are output of the experiment via clayrs
    │   └── 📁 vbpr_cornac                  <- Models which are output of the experiment via cornac
    │
    ├── 📁 reports                       <- Generated metrics and reports by the experiment
    │   ├── 📁 results_clayrs                <- AUC system wise and per user evaluating clayrs models
    │   ├── 📁 results_cornac                <- AUC system wise and per user evaluating cornac models
    │   ├── 📁 ttest_results                 <- Results of the ttest statistic for each epoch
    │   └── 📄 experiment_output.txt         <- Stdout of the terminal which generated committed results
    │
    ├── 📁 src                           <- Source code for use in this project
    │   ├── 📁 data                          <- Scripts to download and generate data
    │   │   ├── 📄 create_interaction_csv.py
    │   │   ├── 📄 dl_raw_sources.py
    │   │   ├── 📄 extract_features_from_source.py
    │   │   └── 📄 train_test_split.py
    │   │
    │   ├── 📁 evaluation                    <- Scripts to evaluate models and compute ttest
    │   │   ├── 📄 compute_auc.py
    │   │   └── 📄 ttest.py
    │   │
    │   ├── 📁 model                         <- Scripts to train models
    │   │   ├── 📄 clayrs_experiment.py
    │   │   └── 📄 cornac_experiment.py
    │   │
    │   ├── 📄 __init__.py                   <- Makes src a Python module
    │   └── 📄 utils.py                      <- Contains utils function for the project
    │
    ├── 📄 LICENSE                       <- MIT License
    ├── 📄 README.md                     <- The top-level README for developers using this project
    ├── 📄 pipeline.py                   <- Script that can be used to reproduce or customize the experiment pipeline
    └── 📄 requirements.txt              <- The requirements file for reproducing the analysis environment (src package)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
