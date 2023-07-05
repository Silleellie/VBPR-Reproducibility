"""
Module used by the `exp2` experiment.

Performs both the Content Analyzer and Recommender System phase of ClayRS.
The Content Analyzer generates the contents, using the caffe reference model with two different pre-processing
configurations, and serializes them.
The Recommender System trains the VBPR algorithm on the previously produced representations.
"""

import os

import clayrs.content_analyzer as ca
from clayrs.utils import Report

from src import INTERIM_DIR, ExperimentConfig, MODEL_DIR, DATA_DIR, YAML_DIR
from src.model import clayrs_recsys
from src.utils import seed_everything

# seed everything
SEED = seed_everything()


def content_analyzer(output_contents_dir: str):
    """
    Performs the Content Analyzer phase of the `exp2` experiment.
    This phase is carried out using the ClayRS framework.
    The representations that will be generated starting from the images for the tradesy items
    use the following techniques:

        * 'caffe': same model as the one used in the VBPR paper (and pre-processing operations suggested for
            the model by the Caffe framework)
        * 'caffe_center_crop': same configuration, but only center crop to 227x227 dimensions is applied as
            pre-processing operation

    Each serialized content will have two different representations, each one associated to the corresponding field.

    A .yml file containing all the specified techniques and their parameters is saved into the `reports/yaml_clayrs`
    directory.

    Args:
        output_contents_dir: path to the directory where the contents will be serialized

    """

    caffe_model_dir = os.path.join(MODEL_DIR, "reference_caffenet")

    prototxt = os.path.join(caffe_model_dir, "deploy.prototxt")
    caffe_model = os.path.join(caffe_model_dir, "bvlc_reference_caffenet.caffemodel")
    mean_pixel = os.path.join(caffe_model_dir, "ilsvrc_2012_mean.npy")

    # pylint: disable=duplicate-code
    tradesy_config = ca.ItemAnalyzerConfig(
        source=ca.CSVFile(os.path.join(INTERIM_DIR, 'tradesy_images_paths.csv')),
        id='itemID',
        output_directory=output_contents_dir
    )

    imgs_dirs = os.path.join(INTERIM_DIR, "imgs_dirs")

    tradesy_config.add_multiple_config(
        'image_path',
        [
            ca.FieldConfig(
                ca.CaffeImageModels(prototxt, caffe_model,
                                    feature_layer='relu7',
                                    mean_file_path=mean_pixel,
                                    batch_size=512,
                                    resize_size=(227, 227),
                                    swap_rb=True,
                                    imgs_dirs=imgs_dirs),
                preprocessing=[
                    ca.TorchLambda(lambda x: x * 255)
                ],
                id='caffe'
            ),

            ca.FieldConfig(
                ca.CaffeImageModels(prototxt, caffe_model,
                                    feature_layer='relu7',
                                    batch_size=512,
                                    resize_size=(300, 300),
                                    imgs_dirs=imgs_dirs),
                preprocessing=[
                    ca.TorchCenterCrop(227),
                    ca.TorchLambda(lambda x: x * 255)
                ],
                id='caffe_center_crop'
            )

        ]
    )

    content_a = ca.ContentAnalyzer(config=tradesy_config, n_thread=ExperimentConfig.num_threads_ca)
    content_a.fit()

    Report(output_dir=YAML_DIR, ca_report_filename="exp2_ca_report").yaml(content_analyzer=content_a)

    print()
    print(f"Output of the Content Analyzer saved into {output_contents_dir}!")
    print(f"Report of the Content Analyzer saved into {os.path.join(YAML_DIR, 'exp2_ca_report.yml')}!")


def main():
    """
    Actual main function of the module.

    It first serializes the contents complexly represented (invoking `content_analyzer()`), and then it
    fits different VBPR algorithms, for 'caffe' and 'caffe_center_crop' representations, using the ClayRS framework
    depending on the number of epochs specified in the `-epo` cmd argument (invoking `clayrs_recsys()`)

    The fit recommenders will be saved into the `models/exp2` directory.

    """

    print("".center(80, "-"))

    output_contents_dir = os.path.join(DATA_DIR, "exp2_ca_output")

    # pylint: disable=duplicate-code
    if not os.path.isdir(output_contents_dir):
        content_analyzer(output_contents_dir)
    else:
        print(f"Serialized contents are already present in {output_contents_dir}, "
              f"content analyzer phase has been skipped")

    print("".center(80, '-'))

    models_dir = os.path.join(MODEL_DIR, "exp2")
    os.makedirs(models_dir, exist_ok=True)

    clayrs_recsys(contents_dir=output_contents_dir,
                  item_field="image_path",
                  field_representation_list=["caffe", "caffe_center_crop"],
                  exp_string="exp2",
                  models_dir=models_dir)
