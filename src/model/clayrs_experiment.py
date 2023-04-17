import os
import pickle

import clayrs_can_see.content_analyzer as ca
import clayrs_can_see.recsys as rs

from src import PROCESSED_DIR, DATA_DIR, MODEL_DIR, ExperimentConfig
from src.utils import load_user_map, load_item_map


def content_analyzer_tradesy(tradesy_item_map_path: str, features_matrix_path: str, output_contents_dir: str):
    items_ca = ca.ItemAnalyzerConfig(
        source=ca.CSVFile(tradesy_item_map_path),
        id="item_id",
        output_directory=output_contents_dir
    )

    items_ca.add_single_config(
        "item_idx",
        ca.FieldConfig(
            ca.FromNPY(
                npy_file_path=features_matrix_path
            )
        )
    )

    ca.ContentAnalyzer(items_ca, n_thread=ExperimentConfig.num_threads).fit()

    print()
    print(f"Output of the Content Analyzer saved into {output_contents_dir}!")


def recsys_tradesy(train_set_path: str, items_dir: str, epoch: int):
    user_map = load_user_map()
    item_map = load_item_map()

    train_set = ca.Ratings(ca.CSVFile(train_set_path), user_map=user_map, item_map=item_map)

    item_field = {'item_idx': 0}

    alg = rs.VBPR(item_field, device='cuda:0',
                  epochs=epoch,
                  gamma_dim=ExperimentConfig.gamma_dim,
                  theta_dim=ExperimentConfig.theta_dim,
                  batch_size=ExperimentConfig.batch_size,
                  learning_rate=ExperimentConfig.lr,
                  seed=ExperimentConfig.random_state,
                  threshold=0,
                  normalize=True)

    cbrs = rs.ContentBasedRS(alg, train_set, items_dir)

    cbrs.fit()

    return cbrs


def main():

    print("".center(80, '-'))

    if not os.path.isdir(os.path.join(DATA_DIR, "clayrs_output_ca")):
        content_analyzer_tradesy(os.path.join(PROCESSED_DIR, "item_map.csv"),
                                 features_matrix_path=os.path.join(PROCESSED_DIR, "features_matrix.npy"),
                                 output_contents_dir=os.path.join(DATA_DIR, "clayrs_output_ca"))
    else:
        print(f"Serialized contents are already present in {os.path.join(DATA_DIR, 'clayrs_output_ca')}, "
              f"content analyzer phase has been skipped")

    os.makedirs(os.path.join(MODEL_DIR, "vbpr_clayrs"), exist_ok=True)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, '-'))
        cbrs = recsys_tradesy(train_set_path=os.path.join(PROCESSED_DIR, "train_set.csv"),
                              items_dir=os.path.join(DATA_DIR, "clayrs_output_ca"),
                              epoch=epoch)

        fname_cbrs = os.path.join(MODEL_DIR, "vbpr_clayrs", f"vbpr_clayrs_{epoch}.ml")

        with open(fname_cbrs, "wb") as f:
            pickle.dump(cbrs, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"ClayRS model for {epoch} epochs saved into {fname_cbrs}!")

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


if __name__ == "__main__":
    main()
