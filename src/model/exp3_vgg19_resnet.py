"""
Module used by the `exp3` experiment.

Performs both the Content Analyzer and Recommender System phase of ClayRS.
The Content Analyzer generates the contents, using the vgg19 and resnet50 pre-trained models, and serializes them.
The Recommender System trains the VBPR algorithm on the previously produced representations.
"""

import os
import torch

from src import INTERIM_DIR, ExperimentConfig, MODEL_DIR, DATA_DIR, YAML_DIR
from src.model import clayrs_recsys
from src.utils import seed_everything

import clayrs_can_see.content_analyzer as ca
from clayrs_can_see.utils import Report

# seed everything
SEED = seed_everything()


def content_analyzer(output_contents_dir: str):
    """
    Performs the Content Analyzer phase of the `exp2` experiment.
    This phase is carried out using the ClayRS framework.
    The representations that will be generated starting from the images for the tradesy items
    use the following techniques:

        * 'vgg19': features are extracted from the last convolution layer before the fully-connected ones
            of the *vgg19* architecture and global max-pooling is applied to them
        * 'resnet50': features are extracted from the *pool5* layer of the *ResNet50* architecture

    Each serialized content will have two different representations, each one associated to the corresponding field.

    A .yml file containing all the specified techniques and their parameters is saved into the `reports/yaml_clayrs`
    directory.

    Args:
        output_contents_dir: path to the directory where the contents will be serialized

    """

    # pylint: disable=duplicate-code
    tradesy_config = ca.ItemAnalyzerConfig(
        source=ca.CSVFile(os.path.join(INTERIM_DIR, 'tradesy_images_paths.csv')),
        id='itemID',
        output_directory=output_contents_dir
    )

    imgs_dirs = os.path.join(INTERIM_DIR, "imgs_dirs")

    # pylint: disable=invalid-name
    def pool_and_squeeze(x: torch.Tensor):
        return torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:]).squeeze()

    tradesy_config.add_multiple_config(
        'image_path',
        [
            ca.FieldConfig(
                ca.PytorchImageModels('vgg19', resize_size=(256, 256), device='cuda:0',
                                      batch_size=32, feature_layer=-3, apply_on_output=pool_and_squeeze,
                                      imgs_dirs=imgs_dirs),
                preprocessing=[
                    ca.TorchCenterCrop(224),
                    ca.TorchNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
                id='vgg19'
            ),

            ca.FieldConfig(
                ca.PytorchImageModels('resnet50', resize_size=(256, 256), device='cuda:0',
                                      batch_size=32, feature_layer=-2, imgs_dirs=imgs_dirs),
                preprocessing=[
                    ca.TorchCenterCrop(224),
                    ca.TorchNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
                id='resnet50'
            ),
        ]
    )

    content_a = ca.ContentAnalyzer(config=tradesy_config, n_thread=ExperimentConfig.num_threads_ca)
    content_a.fit()

    Report(output_dir=YAML_DIR, ca_report_filename="exp3_ca_report").yaml(content_analyzer=content_a)

    print()
    print(f"Output of the Content Analyzer saved into {output_contents_dir}!")
    print(f"Report of the Content Analyzer saved into {os.path.join(YAML_DIR, 'exp3_ca_report.yml')}!")


def main():
    """
    Actual main function of the module.

    It first serializes the contents complexly represented (invoking `content_analyzer()`), and then it
    fits different VBPR algorithms, for 'vgg19' and 'resnet50' representations, using the ClayRS framework
    depending on the number of epochs specified in the `-epo` cmd argument (invoking `clayrs_recsys()`)

    The fit recommenders will be saved into the `models/exp3` directory.

    """

    print("".center(80, "-"))

    output_contents_dir = os.path.join(DATA_DIR, "exp3_ca_output")

    # pylint: disable=duplicate-code
    if not os.path.isdir(output_contents_dir):
        content_analyzer(output_contents_dir)
    else:
        print(f"Serialized contents are already present in {output_contents_dir}, "
              f"content analyzer phase has been skipped")

    print("".center(80, '-'))

    models_dir = os.path.join(MODEL_DIR, "exp3")
    os.makedirs(models_dir, exist_ok=True)

    clayrs_recsys(contents_dir=output_contents_dir,
                  item_field="image_path",
                  field_representation_list=["vgg19", "resnet50"],
                  exp_string="exp3",
                  models_dir=models_dir)
