# VBPR Replicability: comparison and additional experiment

Repository which includes everything needed to replicate VBPR paper by Prof. Julian McAuley of 2016 with a modified version of the ClayRS framework and the original version of the Cornac framework.
It also contains everything to reproduce an end-to-end experiment using the modified version of ClayRS, 
from feature extraction using the *caffe reference model* (with two different pre-processing pipelines) to *resnet50* and *vgg19*.

## How to Use

Simply execute `pip install requirements.txt` in a freshly created *virtual environment*. The source code has been tested with ***python 3.9***.

To perform the 'comparison' experiment between ClayRS and Cornac, run via *command line*:

```
python pipeline.py
```

In this way, raw data will first be *downloaded* and *processed*, and then the actual experiment will be run using the ***default parameters***.
* By default, the experiment is run for $5$, $10$, $20$ and $50$ ***epochs***. Default parameters can be easily changed by passing them as *command line arguments*

To perform the 'additional' experiment using ClayRS, run via *command line*:

```
python pipeline.py -epo 10 20 -exp additional
```

* The experiment was performed by setting 10 and 20 epochs using the `epo` parameter, however any number of epochs can be specified

You can inspect all the parameters that can be set by simply running `python pipeline.py –h`. The following is what you would obtain:

```console
$ python pipeline.py –h

usage: pipeline.py [-h] [-epo 5 [5 ...]] [-bs 128] [-gd 20] [-td 20] [-lr 0.005] [-seed 42] [-exp comparison]

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
  -exp comparison, --experiment comparison
                        Whether to perform the comparison experiment with Cornac, 
                        or the additional one with feature extraction using ClayRS
```

## Experiment pipeline

**TO DO**

### -exp comparison

***Data***:

* Download binary file containing features of images from ....
* Download raw tradesy feedback from ....
* Filtering raw interactions following original VBPR paper instruction and removing duplicate interactions
* Build user map (following the order in which each user appears in the filtered interactions) and item map (following the order in which each item appears in the binary file)
* Extracting into an npy matrix features from the binary file for items which appear in the filtered interactions
* Building train and test set with leave-one-out using `-seed` parameter as random state

***Experiment and evaluation***:

* Fit VBPR algorithm via *ClayRS can see* and *Cornac* using command line arguments when invoking `pipeline.py` (`-epo`, `-bs`, `-gd`, etc.)
* Compute AUC of each user and the average AUC for both *ClayRS* and *Cornac*
* Perform ttest statistical test between *ClayRS* user results and *Cornac* user results

### -exp additional

**TO DO**


Project Organization
------------
    ├── 📁 clayrs_can_see                <- Package containing a modified version of clayrs with VBPR support
    │
    ├── 📁 data                          <- Directory containing all data generated/used by both experiments
    │   ├── 📁 interim                       <- Intermediate data that has been transformed
    │   ├── 📁 processed                     <- The final, canonical data sets used for training
    │   └── 📁 raw                           <- The original, immutable data dump
    │
    ├── 📁 models                        <- Trained and serialized models at different epochs for both experiments
    │   ├── 📁 additional_exp_vbpr           <- Models which are output of the additional experiment via clayrs
    │   ├── 📁 vbpr_clayrs                   <- Models which are output of the comparison experiment via clayrs
    │   └── 📁 vbpr_cornac                   <- Models which are output of the comparison experiment via cornac
    │
    ├── 📁 reports                       <- Generated metrics and reports by both experiments
    │   ├── 📁 results_additional_exp        <- AUC system wise and per user evaluating additional experiment clayrs models
    │   ├── 📁 results_clayrs                <- AUC system wise and per user evaluating comparison experiment clayrs models
    │   ├── 📁 results_cornac                <- AUC system wise and per user evaluating comparison experiment cornac models
    │   ├── 📁 ttest_results                 <- Results of the ttest statistic for each epoch for both experiments
    │   ├── 📄 additional_exp_output.txt     <- Stdout of the additional experiment terminal which generated committed results
    │   └── 📄 comparison_exp_output.txt     <- Stdout of the comparison experiment terminal which generated committed results
    │
    ├── 📁 src                           <- Source code of the project
    │   ├── 📁 data                          <- Scripts to download and generate data
    │   │   ├── 📄 create_interaction_csv.py
    │   │   ├── 📄 create_tradesy_images_dataset.py
    │   │   ├── 📄 dl_raw_sources.py
    │   │   ├── 📄 extract_features_from_source.py
    │   │   └── 📄 train_test_split.py
    │   │
    │   ├── 📁 evaluation                    <- Scripts to evaluate models and compute ttest
    │   │   ├── 📄 compute_auc.py
    │   │   └── 📄 ttest.py
    │   │
    │   ├── 📁 model                         <- Scripts to train models
    │   │   ├── 📄 additional_experiment.py
    │   │   ├── 📄 clayrs_experiment.py
    │   │   └── 📄 cornac_experiment.py
    │   │
    │   ├── 📄 __init__.py                   <- Makes src a Python module
    │   └── 📄 utils.py                      <- Contains utils function for the project
    │
    ├── 📄 LICENSE                       <- MIT License
    ├── 📄 README.md                     <- The top-level README for developers using this project
    ├── 📄 pipeline.py                   <- Script that can be used to reproduce or customize the experiment pipeline
    ├── 📄 requirements.txt              <- The requirements file for reproducing the analysis environment (src package)
    └── 📄 requirements-clayrs.txt       <- The requirements file for the modified version of clayrs

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
