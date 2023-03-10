# VBPR-replicability
Repository which includes all things related to replicate VBPR paper by Prof. Julian McAuley of 2016


Project Organization
------------
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
