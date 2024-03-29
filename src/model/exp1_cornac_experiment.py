"""
Module used by the `exp1` experiment.

Uses the NPY feature matrix built in the data preparation phase to fit the VBPR algorithm using the Cornac framework.
"""

import copy
import os
import pickle

import cornac
import numpy as np
import torch

from cornac.data.dataset import Dataset
from cornac.data import ImageModality

from src import PROCESSED_DIR, MODEL_DIR, ExperimentConfig
from src.utils import load_user_map, load_train_test_instances, load_item_map, seed_everything

# seed everything
SEED = seed_everything()


def build_train(feature_matrix_path: str):
    """
    Build the train Dataset data structure required by the Cornac library using serialized train set,
    user map and item map and NPY feature matrix.

    Args:
        feature_matrix_path: path where the .npy matrix containing visual features is stored

    Returns:
        train_dataset: built Cornac Dataset data structure with item visual features as image modality

    """

    features = np.load(feature_matrix_path)

    train_tuples = load_train_test_instances(mode="train")

    user_map = load_user_map()
    item_map = load_item_map()

    train_dataset = Dataset.build(train_tuples, global_uid_map=user_map, global_iid_map=item_map, seed=SEED)

    # mock iterator to disable shuffle for replicability
    train_dataset.uij_iter = lambda batch_size, shuffle: Dataset.uij_iter(train_dataset, batch_size, shuffle=False)

    # Instantiate a ImageModality for the two datasets
    item_image_modality = ImageModality(features=features, ids=list(item_map.keys()), normalized=True)
    item_image_modality.build()

    train_dataset.add_modalities(item_image=item_image_modality)

    return train_dataset, features


def train_cornac(train_dataset: Dataset, features: np.ndarray, epoch: int):
    """
    Train a VBPR Recommender using the Cornac framework

    Args:
        train_dataset: cornac Dataset data structure representing train set
        features: numpy array containing visual features for each item (each row represents an item)
        epoch: number of epochs to train the model for

    Returns:
        vbpr: Trained VBPR Recommender from the cornac framework

    """

    # Init parameters since cornac uses numpy and ClayRS uses torch for initialization,
    # torch and numpy uses different seeds
    gamma_dim = ExperimentConfig.gamma_dim
    theta_dim = ExperimentConfig.theta_dim
    features_dim = features.shape[1]

    n_users = train_dataset.total_users
    n_items = train_dataset.total_items

    Gu = torch.zeros(size=(n_users, gamma_dim))  # pylint: disable=invalid-name
    Gi = torch.zeros(size=(n_items, gamma_dim))  # pylint: disable=invalid-name

    Tu = torch.zeros(size=(n_users, theta_dim))  # pylint: disable=invalid-name

    E = torch.zeros(size=(features_dim, theta_dim))  # pylint: disable=invalid-name
    Bp = torch.zeros(size=(features_dim, 1))  # pylint: disable=invalid-name

    Bi = torch.zeros(size=(n_items, 1)).squeeze()  # pylint: disable=invalid-name

    # seed torch
    torch.manual_seed(SEED)

    torch.nn.init.xavier_uniform_(Gu)
    torch.nn.init.xavier_uniform_(Gi)
    torch.nn.init.xavier_uniform_(Tu)
    torch.nn.init.xavier_uniform_(E)
    torch.nn.init.xavier_uniform_(Bp)

    # Instantiate VBPR
    vbpr = cornac.models.VBPR(
        k=gamma_dim,
        k2=theta_dim,
        n_epochs=epoch,
        batch_size=ExperimentConfig.batch_size,
        learning_rate=ExperimentConfig.lr,
        lambda_w=0.01,
        lambda_b=0.01,
        lambda_e=0.0,
        use_gpu=True,
        seed=SEED,
        init_params={
            "Gu": Gu.numpy(),
            "Gi": Gi.numpy(),
            "Tu": Tu.numpy(),
            "E": E.numpy(),
            "Bp": Bp.numpy(),
            "Bi": Bi.numpy()
        }
    )

    vbpr.fit(train_dataset)

    return vbpr


def main():
    """
    Actual main function of the module.

    It first builds the train Dataset data structure required by the Cornac library (invoking `build_train()`),
    and then fits the VBPR algorithm using the same library on the number of epochs specified in the `-epo` cmd argument

    The fit algorithms will be saved into the `models/exp1/vbpr_cornac` directory.

    """
    models_dir = os.path.join(MODEL_DIR, "exp1", "vbpr_cornac")
    os.makedirs(models_dir, exist_ok=True)

    feature_matrix_path = os.path.join(PROCESSED_DIR, "features_matrix.npy")

    train_set, features = build_train(feature_matrix_path)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, '-'))
        vbpr = train_cornac(train_set, features, epoch)

        fname = os.path.join(models_dir, f"vbpr_cornac_{epoch}.ml")

        with open(fname, "wb") as file:
            pickle.dump(copy.deepcopy(vbpr), file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Cornac model for {epoch} epochs saved into {fname}!")

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


if __name__ == "__main__":
    main()
