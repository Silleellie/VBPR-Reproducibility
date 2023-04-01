import os
import pickle
import random
import torch
import numpy as np

from src import INTERIM_DIR, PROCESSED_DIR, ExperimentConfig, MODEL_DIR, DATA_DIR

import clayrs_can_see.content_analyzer as ca
import clayrs_can_see.recsys as rs
from src.utils import load_user_map, load_item_map

# seed everything
seed = ExperimentConfig.random_state
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
print(f"Random seed set as {seed}")


def content_analyzer(output_contents_dir):

    caffe_model_dir = os.path.join(MODEL_DIR, "reference_caffenet")

    prototxt = os.path.join(caffe_model_dir, "deploy.prototxt")
    caffe_model = os.path.join(caffe_model_dir, "bvlc_reference_caffenet.caffemodel")
    mean_pixel = os.path.join(caffe_model_dir, "ilsvrc_2012_mean.npy")

    tradesy_config = ca.ItemAnalyzerConfig(
        source=ca.CSVFile(os.path.join(INTERIM_DIR, 'tradesy_images_paths.csv')),
        id='itemID',
        output_directory=output_contents_dir
    )

    tradesy_config.add_multiple_config(
        'image_path',
        [

            ca.FieldConfig(
                ca.PytorchImageModels('resnet50', resize_size=(256, 256), device='cuda:0', batch_size=32, feature_layer=-2),
                preprocessing=[
                    ca.TorchCenterCrop(224),
                    ca.TorchNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
                id='resnet50'
            ),

            ca.FieldConfig(
                ca.CaffeImageModels(prototxt, caffe_model,
                                    feature_layer='relu7',
                                    mean_file_path=mean_pixel,
                                    batch_size=512,
                                    resize_size=(227, 227),
                                    swapRB=True),
                preprocessing=[
                    ca.TorchLambda(lambda x: x * 255)
                ],
                id='caffe'
            ),

            ca.FieldConfig(
                ca.CaffeImageModels(prototxt, caffe_model,
                                    feature_layer='relu7',
                                    batch_size=512,
                                    resize_size=(300, 300)),
                preprocessing=[
                    ca.TorchCenterCrop(227),
                    ca.TorchLambda(lambda x: x * 255)
                ],
                id='caffe_center_crop'
            )

        ]
    )

    ca.ContentAnalyzer(config=tradesy_config).fit()

    print()
    print(f"Output of the Content Analyzer saved into {output_contents_dir}!")


def recommender_system(contents_dir):

    models_dir = os.path.join(MODEL_DIR, "additional_exp_vbpr")
    os.makedirs(models_dir, exist_ok=True)

    user_map = load_user_map()
    item_map = load_item_map()

    train_set = ca.Ratings(ca.CSVFile(os.path.join(PROCESSED_DIR, "train_set.csv")), user_map=user_map,
                           item_map=item_map)

    item_fields = [{'image_path': 'resnet50'}, {'image_path': 'caffe'}, {'image_path': 'caffe_center_crop'}]

    for item_field in item_fields:

        print("Performing VBPR training for representation: ", item_field['image_path'])

        for epoch_num in ExperimentConfig.epochs:

            print(f"Considering number of epochs {epoch_num}")
            print("".center(80, '-'))

            alg = rs.VBPR(item_field, device='cuda:0',
                          epochs=epoch_num,
                          gamma_dim=ExperimentConfig.gamma_dim,
                          theta_dim=ExperimentConfig.theta_dim,
                          batch_size=ExperimentConfig.batch_size,
                          learning_rate=ExperimentConfig.lr,
                          seed=ExperimentConfig.random_state,
                          threshold=0,
                          normalize=True)

            cbrs = rs.ContentBasedRS(alg, train_set, contents_dir)

            cbrs.fit()

            fname_cbrs = os.path.join(models_dir, f"additional_exp_{item_field['image_path'][0]}_{epoch_num}.ml")

            with open(fname_cbrs, "wb") as f:
                pickle.dump(cbrs, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"ClayRS model for {epoch_num} epochs saved into {fname_cbrs}!")

            if epoch_num != ExperimentConfig.epochs[-1]:
                print("".center(80, '-'))


def main():

    print("".center(80, "-"))

    output_contents_dir = os.path.join(DATA_DIR, "additional_exp_ca_output")

    if not os.path.isdir(output_contents_dir):
        content_analyzer(output_contents_dir)
    else:
        print(f"Serialized contents are already present in {output_contents_dir}, "
              f"content analyzer phase has been skipped")

    print("".center(80, '-'))

    recommender_system(output_contents_dir)


if __name__ == "__main__":
    main()
