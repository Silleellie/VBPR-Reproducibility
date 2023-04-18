# VBPR Replicability: comparison and additional experiment

![pylint](https://img.shields.io/badge/pylint-10.00-brightgreen?logo=python&logoColor=white)

Repository which includes everything needed to reproduce the VBPR paper by Prof. Julian McAuley of 2016 with a modified version of the ClayRS framework and the original version of the Cornac framework.
It also contains everything to reproduce an end-to-end experiment using the modified version of ClayRS, 
from feature extraction using the *caffe reference model* (with two different pre-processing pipelines) to *resnet50* and *vgg19*.

Check the ['Experiment pipeline' section](#experiment-pipeline) for an overview of the operations carried out by the two different experiments

## How to Use

Simply execute `pip install requirements.txt` in a freshly created *virtual environment*.

The source code has been tested with ***python 3.9*** and **CUDA is required** to run the experiments.

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
  -nt 4, --num_threads 4
                        Number of threads that will be used in ClayRS during Content Analyzer serialization phase
  -exp comparison, --experiment comparison
                        Whether to perform the comparison experiment with Cornac, 
                        or the additional one with feature extraction using ClayRS
```

## Experiment pipeline

The following is a description of the operations carried out by the pipeline depending on the experiment type (additional or comparison)

### -exp comparison

***Data***:

* Download raw tradesy feedback from [here](http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz)
* Download binary file containing features of images from [here](http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b)
* Filter raw interactions following original VBPR paper instructions and remove duplicate interactions
* Build user map (following the order in which each user appears in the filtered interactions) and item map (following the order in which each item appears in the binary file)
* Extract into a npy matrix features from the binary file for items which appear in the filtered interactions
* Build train and test set with leave-one-out using `-seed` parameter as random state

***Experiment and evaluation***:

* Fit VBPR algorithm via *ClayRS can see* and *Cornac* using command line arguments when invoking `pipeline.py` (`-epo`, `-bs`, `-gd`, etc.)
* Compute AUC of each user and the average AUC for both *ClayRS* and *Cornac*
* Perform ttest statistical test between *ClayRS* user results and *Cornac* user results

### -exp additional

***Data***:

* Download raw tradesy feedback from [here](http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz)
* Download npy file containing tradesy images from [here](http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy)
* Download caffe model and all of its necessary files:
  * *bvlc_reference_caffenet model* from [here](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)
  * *deploy.prototxt* for bvlc_reference_caffenet from [here](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt)
  * *ilsvrc_2012_mean.npy* file containing mean pixel value from [here](https://github.com/facebookarchive/models/raw/master/bvlc_reference_caffenet/ilsvrc_2012_mean.npy)
* Filter raw interactions following original VBPR paper instructions and remove duplicate interactions
* Download binary file containing features of images from [here](http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b)
* Build item map (following the order in which each item appears in the binary file)
* Extract from the npy matrix into a folder the images of the items which appear in the filtered interactions
* Build a .csv file associating each item to the path of its image in said directory
* Build train and test set with leave-one-out using `-seed` parameter as random state

***Experiment and evaluation***:

* From the images dataset, create processed contents using the Content Analyzer. Each serialized content (corresponding to an item) 
will have 4 different representations:
  * **caffe**: same model as the one used in the VBPR paper (and pre-processing operations suggested for the model by the Caffe framework from [here](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb))
  * **caffe_center_crop**: same configuration, but only center crop to 227x227 dimensions is applied as pre-processing operation
  * **resnet50**: features are extracted from the *pool5* layer of the *ResNet50* architecture
  * **vgg19**: features are extracted from the last convolution layer before the fully-connected ones  of the *vgg19* architecture and global max-pooling is applied to them
* Fit a different VBPR algorithm for each representation via *ClayRS can see* using command line arguments when invoking `pipeline.py` (`-epo`, `-bs`, `-gd`, etc.)
* Compute AUC of each user and the average AUC for *ClayRS* for each VBPR algorithm instance
* Perform ttest statistical test between each configuration


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
    │   │
    │   ├── 📁 yaml_clayrs                   <- Reports generated by the Report class in ClayRS to document all techniques and parameters used in the experiment
    │   │   ├── 📁 rs_report_additional_exp      <- Reports generated for each Recommender System configuration in the additional experiment
    │   │   ├── 📁 rs_report_comparison_exp      <- Reports generated for each Recommender System configuration in the comparison experiment
    │   │   ├── 📄 ca_report_additional_exp.yml  <- Report generated for the Content Analyzer module in the additional experiment
    │   │   └── 📄 ca_report_additional_exp.yml  <- Report generated for the Content Analyzer module in the comparison experiment
    │   │
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
