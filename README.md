# VBPR Replicability: comparison and end-to-end experiments with ClayRS can see 

![pylint](https://img.shields.io/badge/pylint-10.00-brightgreen?logo=python&logoColor=white)

Repository which includes everything related to the paper ***Reproducibility Analysis of Recommender Systems relying on 
Visual Features: traps, pitfalls, and countermeasures***

The following are the experiments that could be reproduced using this repository:

* Experiment 1: comparing VBPR results
  * *Comparing the implementation of the VBPR algorithm between the modified version of ClayRS and Cornac*
* Experiment 2: Testing ClayRS Can See functionalities to include images as side information
  * *Performing an end-to-end experiment using the modified version of ClayRS with the pre-trained caffe reference 
  model on different pre-processing configurations*
* Experiment 3: Testing state-of-the-art models for extracting features from images
  * *Performing an end-to-end experiment using the modified version of ClayRS with the pre-trained vgg19 and resnet50 
  models*

Check the ['Experiment pipeline' section](#experiment-pipeline) for an overview of the operations carried out by the three different experiments

All the experiments provided in this repository are compliant with the proposed checklist:

<table align="center">
  <thead>
    <tr style="text-align: right;">
      <th>Stage</th>
      <th>Check</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="center">Dataset Collection</td>
      <td>✅ Link to a downloadable version of the dataset collection</td>
      <td>
        <a href="https://drive.google.com/uc?id=1xaRS4qqGeTzxaEksHzjVKjQ6l7QT9eMJ">Tradesy raw feedback</a>,<br>
        <a href="http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b">Image features binary file</a>,<br>
        <a href="http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy">Tradesy Images from DVBPR dataset</a>
      </td>
    </tr>
    <tr>
      <td>✅ Any pre-filtering process performed on data</td>
      <td>
        $\forall$ experiment, duplicate interactions are removed and users with less than five interactions are not considered, <a href="src/data/create_interaction_csv.py">script</a>.<br>
        For <i>Experiment 2</i> and <i>Experiment 3</i>, images from the <i>Tradesy Images DVBPR</i> dataset were removed in order to
        re-create the VBPR dataset (since original dataset is not accessible), <a href="src/data/create_tradesy_images_dataset.py">script</a>
      </td>
    </tr>
    <tr>
      <td>✅ Relevant dataset statistics</td>
      <td>$\forall$ experiment, lines <b>18-27</b> of <a href="reports/exp1_terminal_output.txt">terminal output</a></td>
    </tr>
    <tr>
      <td>✅ Preprocessing operations performed on side information</td>
      <td>
        <i>Experiment 1</i>: no preprocessing performed, visual features provided by original authors were used,<br>
        <i>Experiment 2</i>: lines <b>23-24</b>, <b>42-47</b> of <a href="reports/yaml_clayrs/exp2_ca_report.yml">yaml report</a>, lines <b>71-73</b>, <b>83-86</b> of <a href="src/model/exp2_caffe.py">script</a>,<br>
        <i>Experiment 3</i>: lines <b>21-34</b>, <b>50-63</b> of <a href="reports/yaml_clayrs/exp3_ca_report.yml">yaml report</a>, lines <b>64-67</b>, <b>74-77</b> of <a href="src/model/exp3_vgg19_resnet.py">script</a>
      </td>
    </tr>
    <tr>
      <td>✅ Pre-trained models adopted to represent side information</td>
      <td>
        <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet">bvlc_reference_caffenet</a>,<br>
        <a href="https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/">resnet50</a>,<br>
        <a href="https://pytorch.org/hub/pytorch_vision_vgg/">vgg19</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2" valign="center">Data Splitting</td>
      <td>✅ Protocol used for data partitioning and random seed to reproduce random splits</td>
      <td> <i>Leave-one-out</i> with random seed set at <b>42</b>, <a href="src/data/train_test_split.py">script</a></td>
    </tr>
    <tr>
      <td>⬜ Link to a downloadable version of the training/test/validation sets</td>
      <td> <i>Train</i> and <i>test sets</i> are not provided, but can be easily reproduced by running the <a href="src/data/__main__.py"> main data pipeline </a>,
          by setting the random state to <b>42</b>
      </td>
    </tr>
    <tr>
      <td rowspan="4" valign="center">Recommendation</td>
      <td>✅ Name and version of the framework containing the recommendation algorithm</td>
      <td>
        Clayrs can See (modified version of <a href="https://github.com/swapUniba/ClayRS/releases/tag/v0.4.0"><i>Clayrs v0.4</i></a>),<br>
        <a href="https://github.com/PreferredAI/cornac/releases/tag/v1.14.2"><i>Cornac v1.14.2</i></a>
      </td>
    </tr>
    <tr>
      <td>✅ Source code of the recommendation algorithm and setting of parameters</td>
      <td>
        Source code of the recommendation algorithm:<br>
        <a href="https://github.com/swapUniba/ClayRS/blob/v0.5.1/clayrs/recsys/visual_based_algorithm/vbpr/vbpr_algorithm.py">Clayrs can See VBPR</a>,<br>
        <a href="https://github.com/PreferredAI/cornac/tree/5caf11cffb862c304e4dcc3e0e90c8bdcdc08093/cornac/models/vbpr">Cornac VBPR</a><br>
        <br>
        Parameters settings:<br>
        ClayRS can See: lines <b>61-70</b> of <a href="src/model/__init__.py">script</a>,<br>
        Cornac: lines <b>102-121</b> of <a href="src/model/exp1_cornac_experiment.py">script</a>
      </td>
    </tr>
    <tr>
      <td>⬜ Method to select the best hyperparameters</td>
      <td> No <i>hyperparameter tuning</i> was carried out </td>
    </tr>
    <tr>
      <td>✅ Any random seed necessary to reproduce random processes</td>
      <td> All random processes were set to random seed <b>42</b> </td>
    </tr>
    <tr>
      <td rowspan="2" valign="center">Candidate Item Filtering</td>
      <td>✅ Set of target items to generate a ranking</td>
      <td>
        All items of the system were taken into account
      </td>
    </tr>
    <tr>
      <td>✅ Strategy (TestRatings, TestItems, TrainingItems, AllItems, One-Plus-Random)</td>
      <td>
        <i>AllItems</i>
      </td>
    </tr>
    <tr>
      <td rowspan="5" valign="center">Evaluation</td>
      <td>✅ Name and version of the framework used to compute metrics</td>
      <td>
        Cornac framework for evaluating cornac models,<br>
        <a href="src/evaluation/compute_auc.py">Custom AUC implementation</a> to evaluate ClayRS model, lines of script: <b>64-118</b>
      </td>
    </tr>
    <tr>
      <td>✅ List of metrics adopted and cutoff for recommendation lists</td>
      <td>
        The only metric used was <b>AUC</b>, and all ranked items were taken into account to compute it
      </td>
    </tr>
    <tr>
      <td>⬜ Normalization strategy adopted</td>
      <td>
        No normalization strategy was applied for the metric chosen (<i>AUC</i>)
      </td>
    </tr>
    <tr>
      <td>✅ Averaging strategy adopted (e.g. micro or macro-average)</td>
      <td>
        System results were generated by performing macro-average over the user results,
        line 115 of <a href="src/evaluation/compute_auc.py">script</a>
      </td>
    </tr>
    <tr>
      <td>✅ List of results in a standard format (per fold and overall)</td>
      <td>
        <i>Experiment 1</i> AUC results path: <code>reports/exp1</code>,<br>
        <i>Experiment 2</i> AUC results path: <code>reports/exp2</code>,<br>
        <i>Experiment 3</i> AUC results path: <code>reports/exp3</code>
      </td>
    </tr>
    <tr>
      <td rowspan="3" valign="center">Statistical testing</td>
      <td>✅ Data on which the test is performed</td>
      <td>
        <i>Experiment 1</i>: AUC results between ClayRS and Cornac for each epoch located at <code>reports/exp1</code>,<br>
        <i>Experiment 2</i>: AUC results between caffe and caffe_center_crop trained recommender for each epoch located at <code>reports/exp2</code>,</a><br>
        <i>Experiment 3</i>: AUC results between vgg19 and resnet50 trained recommender for each epoch located at <code>reports/exp3</code></a>
      </td>
    </tr>
    <tr>
      <td>✅ Type of test and p-value</td>
      <td>
        <b>ttest statistical test</b> was used:<br>
        <i>Experiment 1</i> p-value results path: <code>reports/ttest_results/exp1</code>,<br>
        <i>Experiment 2</i> p-value results path: <code>reports/ttest_results/exp2</code>,<br>
        <i>Experiment 3</i> p-value results path: <code>reports/ttest_results/exp3</code>
      </td>
    </tr>
    <tr>
      <td>⬜ Corrections for multiple comparisons</td>
      <td>No correction was applied</td>
    </tr>
  </tbody>
</table>


## How to Use

Simply execute `pip install requirements.txt` in a freshly created *virtual environment*.

The source code has been tested and results have been produced with ***python 3.9*** and ***CUDA V11.6***.
Please note that *CUDA* must be installed to run the experiments.

To perform the `exp1` experiment, which is the comparison of the VBPR implementation between ClayRS and Cornac, 
run via *command line*:

```
python pipeline.py -epo 5 10 20 50 -exp exp1
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
    └── 📄 requirements.txt              <- The requirements file for reproducing the analysis environment (src package)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
