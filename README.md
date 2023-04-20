# VBPR Replicability: comparison and end-to-end experiments with ClayRS can see 

![pylint](https://img.shields.io/badge/pylint-10.00-brightgreen?logo=python&logoColor=white)

Repository which includes everything needed to reproduce the VBPR paper by Prof. Julian McAuley of 2016 with a modified version of the ClayRS framework and the original version of the Cornac framework.
It also contains everything to reproduce two end-to-end experiments using the modified version of ClayRS, 
from feature extraction using the *caffe reference model* (with two different pre-processing pipelines) to *resnet50* and *vgg19*.

Check the ['Experiment pipeline' section](#experiment-pipeline) for an overview of the operations carried out by the three different experiments

## How to Use

Simply execute `pip install requirements.txt` in a freshly created *virtual environment*.

The source code has been tested with ***python 3.9*** and **CUDA is required** to run the experiments.

To perform the `exp1` experiment, which is the comparison of the VBPR implementation between ClayRS and Cornac, 
run via *command line*:

```
python pipeline.py
```

In this way, raw data will first be *downloaded* and *processed*, and then the actual experiment will be run using the ***default parameters***.
* By default, the experiment is run for $5$, $10$, $20$ and $50$ ***epochs***. Default parameters can be easily changed by passing them as *command line arguments*

To perform the `exp2` experiment, which is the end-to-end experiment in which ClayRS can see is tested to include
images as side information (using *bvlc_reference_caffenet* with two different pre-processing configurations), run via *command line*:

```
python pipeline.py -epo 10 20 -exp exp2
```

* The experiment was performed by setting 10 and 20 epochs using the `epo` parameter, however any number of epochs can be specified

To perform the `exp3` experiment, which is the end-to-end experiment in which ClayRS can see is tested using
state-of-the-art models (*vgg19* and *resnet50*) for extracting features from images, run via *command line*:

```
python pipeline.py -epo 10 20 -exp exp3
```

You can inspect all the parameters that can be set by simply running `python pipeline.py –h`. The following is what you would obtain:

```console
$ python pipeline.py –h

usage: pipeline.py [-h] [-epo 5 [5 ...]] [-bs 128] [-gd 20] [-td 20] [-lr 0.005] [-seed 42] [-nt_ca 4] [-exp exp1]

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
  -nt_ca 4, --num_threads_ca 4
                        Number of threads that will be used in ClayRS during Content Analyzer serialization phase
  -exp exp1, --experiment exp1
                        exp1 to perform the comparison experiment with Cornac,
                        exp2 to perform end to end experiment using caffe via ClayRS can see,
                        exp3 to perform end to end experiment using vgg19 and resnet50 via Clayrs can see
```

## Experiment pipeline

The following is a description of the operations carried out by the pipeline depending on the experiment type
(`exp1`, `exp2`, `exp3`) set by changing the `-exp` parameter

### -exp exp1

***Data***:

* Download raw tradesy feedback from [here](http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz)
* Download binary file containing features of images from [here](http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b)
* Filter raw interactions following original VBPR paper instructions and remove duplicate interactions
* Extract into a npy matrix features from the binary file for items which appear in the filtered interactions
* Build item map (following the order in which each item appears in the binary file)
* Build train and test set with leave-one-out using `-seed` parameter as random state
* Build user map (following the order in which each user appears in the filtered interactions)

***Experiment and evaluation***:

* Fit VBPR algorithm via *ClayRS can see* and *Cornac* using command line arguments when invoking `pipeline.py` (`-epo`, `-bs`, `-gd`, etc.)
* Compute AUC of each user and the average AUC for both *ClayRS* and *Cornac*
* Perform ttest statistical test between *ClayRS* user results and *Cornac* user results

### -exp exp2

***Data***:

* Download raw tradesy feedback from [here](http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz)
* Download npy file containing tradesy images from [here](http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy)
* Download caffe model and all of its necessary files:
  * *bvlc_reference_caffenet model* from [here](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)
  * *deploy.prototxt* for bvlc_reference_caffenet from [here](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt)
  * *ilsvrc_2012_mean.npy* file containing mean pixel value from [here](https://github.com/facebookarchive/models/raw/master/bvlc_reference_caffenet/ilsvrc_2012_mean.npy)
* Filter raw interactions following original VBPR paper instructions and remove duplicate interactions
* Download binary file containing features of images from [here](http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b)
* Extract into a npy matrix features from the binary file for items which appear in the filtered interactions
* Build item map (following the order in which each item appears in the binary file)
* Extract from the npy file into a folder the images of the items which appear in the filtered interactions
* Build a .csv file associating each item to the path of its image in said directory
* Build train and test set with leave-one-out using `-seed` parameter as random state
* Build user map (following the order in which each user appears in the filtered interactions)

***Experiment and evaluation***:

* From the images dataset, create processed contents using the Content Analyzer. Each serialized content (corresponding to an item) 
will have two different representations:
  * **caffe**: same model as the one used in the VBPR paper (and pre-processing operations suggested for the model by the Caffe framework from [here](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb))
  * **caffe_center_crop**: same configuration, but only center crop to 227x227 dimensions is applied as pre-processing operation
* Fit a different VBPR algorithm for the two representations via *ClayRS can see* using command line arguments when invoking `pipeline.py` (`-epo`, `-bs`, `-gd`, etc.)
* Compute AUC of each user and the average AUC for *ClayRS* for each VBPR algorithm instance
* Perform ttest statistical test between the two configurations

### -exp exp3

***Data***:

* Download raw tradesy feedback from [here](http://jmcauley.ucsd.edu/data/tradesy/tradesy.json.gz)
* Download npy file containing tradesy images from [here](http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy)
* Filter raw interactions following original VBPR paper instructions and remove duplicate interactions
* Download binary file containing features of images from [here](http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b)
* Extract into a npy matrix features from the binary file for items which appear in the filtered interactions
* Build item map (following the order in which each item appears in the binary file)
* Extract from the npy file into a folder the images of the items which appear in the filtered interactions
* Build a .csv file associating each item to the path of its image in said directory
* Build train and test set with leave-one-out using `-seed` parameter as random state
* Build user map (following the order in which each user appears in the filtered interactions)

***Experiment and evaluation***:

* From the images dataset, create processed contents using the Content Analyzer. Each serialized content (corresponding to an item) 
will have two different representations:
  * **resnet50**: features are extracted from the *pool5* layer of the *ResNet50* architecture
  * **vgg19**: features are extracted from the last convolution layer before the fully-connected ones  of the *vgg19* architecture and global max-pooling is applied to them
* Fit a different VBPR algorithm for the two representations via *ClayRS can see* using command line arguments when invoking `pipeline.py` (`-epo`, `-bs`, `-gd`, etc.)
* Compute AUC of each user and the average AUC for *ClayRS* for each VBPR algorithm instance
* Perform ttest statistical test between the two configurations


Project Organization
------------
    ├── 📁 clayrs_can_see                <- Package containing a modified version of clayrs with VBPR support
    │
    ├── 📁 data                          <- Directory containing all data generated/used by both experiments
    │   ├── 📁 interim                       <- Intermediate data that has been transformed
    │   ├── 📁 processed                     <- The final, canonical data sets used for training
    │   └── 📁 raw                           <- The original, immutable data dump
    │
    ├── 📁 models                        <- Trained and serialized models at different epochs for the three experiments
    │   ├── 📁 exp1                          <- Models which are output of the experiment 1
    │   │   ├── 📁 vbpr_clayrs                   <- ClayRS models which are output of the experiment 1
    │   │   └── 📁 vbpr_cornac                   <- Cornac models which are output of the experiment 1
    │   │
    │   ├── 📁 exp2                          <- Models which are output of the experiment 2
    │   └── 📁 exp3                          <- Models which are output of the experiment 3
    │
    ├── 📁 reports                       <- Generated metrics and reports by the three different experiments
    │   ├── 📁 exp1                          <- System-wise and per-user AUC results output of the experiment 1
    │   │   ├── 📁 vbpr_clayrs                   <- ClayRS AUC results which are output of the experiment 1
    │   │   └── 📁 vbpr_cornac                   <- Cornac AUC results which are output of the experiment 1
    │   │
    │   ├── 📁 exp2                          <- System-wise and per-user AUC results output of the experiment 2
    │   ├── 📁 exp3                          <- System-wise and per-user AUC results output of the experiment 3
    │   ├── 📁 ttest_results                 <- Results of the ttest statistic for each epoch for all three experiments
    │   │   ├── 📁 exp1                          <- ttest results output of the experiment 1
    │   │   ├── 📁 exp2                          <- ttest results output of the experiment 2
    │   │   └── 📁 exp3                          <- ttest results output of the experiment 3
    │   │
    │   ├── 📁 yaml_clayrs                   <- Reports generated by the Report class in ClayRS to document all techniques and parameters used in the experiments
    │   │   ├── 📁 exp1_rs_report                <- Reports generated for each Recommender System configuration in the experiment 1
    │   │   ├── 📁 exp2_rs_report                <- Reports generated for each Recommender System configuration in the experiment 2
    │   │   ├── 📁 exp3_rs_report                <- Reports generated for each Recommender System configuration in the experiment 3
    │   │   ├── 📄 exp1_ca_report.yml            <- Report generated for the Content Analyzer module in the experiment 1
    │   │   ├── 📄 exp2_ca_report.yml            <- Report generated for the Content Analyzer module in the experiment 2
    │   │   └── 📄 exp3_ca_report.yml            <- Report generated for the Content Analyzer module in the experiment 3
    │   │
    │   ├── 📄 exp1_terminal_output.txt      <- Output of the terminal which generated committed results for experiment 1
    │   ├── 📄 exp2_terminal_output.txt      <- Output of the terminal which generated committed results for experiment 2
    │   └── 📄 exp3_terminal_output.txt      <- Output of the terminal which generated committed results for experiment 3
    │
    ├── 📁 src                           <- Source code of the project
    │   ├── 📁 data                          <- Scripts to download and generate data
    │   │   ├── 📄 create_interaction_csv.py
    │   │   ├── 📄 create_tradesy_images_dataset.py
    │   │   ├── 📄 dl_raw_sources.py
    │   │   ├── 📄 extract_features_from_source.py
    │   │   └── 📄 train_test_split.py
    │   │
    │   ├── 📁 evaluation                <- Scripts to evaluate models and compute ttest
    │   │   ├── 📄 compute_auc.py
    │   │   └── 📄 ttest.py
    │   │
    │   ├── 📁 model                     <- Scripts to train models
    │   │   ├── 📄 exp1_clayrs_experiment.py
    │   │   ├── 📄 exp1_cornac_experiment.py
    │   │   ├── 📄 exp2_caffe.py
    │   │   ├── 📄 exp3_vgg19_resnet.py
    │   │   ├── 📄 clayrs_experiment.py
    │   │   └── 📄 cornac_experiment.py
    │   │
    │   ├── 📄 __init__.py                   <- Makes src a Python module
    │   └── 📄 utils.py                      <- Contains utils function for the project
    │
    ├── 📄 LICENSE                       <- MIT License
    ├── 📄 pipeline.py                   <- Script that can be used to reproduce or customize the experiment pipeline
    ├── 📄 README.md                     <- The top-level README for developers using this project
    ├── 📄 requirements.txt              <- The requirements file for reproducing the analysis environment (src package)
    └── 📄 requirements-clayrs.txt       <- The requirements file for the modified version of clayrs

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
