import os
import pickle

from src import PROCESSED_DIR, ExperimentConfig, YAML_DIR
from src.utils import load_user_map, load_item_map

import clayrs_can_see.content_analyzer as ca
import clayrs_can_see.recsys as rs
from clayrs_can_see.utils import Report


def clayrs_recsys(contents_dir: str,
                  item_field: str,
                  field_representation_list: list,
                  exp_string: str,
                  models_dir: str):
    """
    Performs the Recommender System phase of the `additional` experiment.

    Trains a recommender system using the VBPR algorithm via the ClayRS framework, one for each representation created
    by the Content Analyzer. Each fit recommender is then saved in the `models/additional_exp_vbpr` directory.

    A .yml file containing the VBPR algorithm definition with its parameters is saved into the
    `reports/yaml_clayrs/rs_report_additional_exp` directory.

    Args:
        contents_dir: path to the directory where the serialized contents are stored

    """
    os.makedirs(os.path.join(YAML_DIR, f"{exp_string}_rs_report"), exist_ok=True)

    user_map = load_user_map()
    item_map = load_item_map()

    train_set = ca.Ratings(ca.CSVFile(os.path.join(PROCESSED_DIR, "train_set.csv")),
                           user_map=user_map,
                           item_map=item_map)

    print("".center(80, "*"))
    for representation_id in field_representation_list:

        print("Performing VBPR training for representation: ", representation_id)
        print("".center(80, "*"))

        for epoch_num in ExperimentConfig.epochs:

            print(f"Considering number of epochs {epoch_num}")
            print("".center(80, '-'))

            alg = rs.VBPR({item_field: representation_id},
                          device='cuda:0',
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

            fname_cbrs = os.path.join(models_dir, f"vbpr_clayrs_{representation_id}_{epoch_num}.ml")

            with open(fname_cbrs, "wb") as file:
                pickle.dump(cbrs, file, protocol=pickle.HIGHEST_PROTOCOL)

            Report(output_dir=os.path.join(YAML_DIR, f"{exp_string}_rs_report"),
                   rs_report_filename=f"rs_report_{representation_id}_{epoch_num}").yaml(recsys=cbrs)

            output_report_path = os.path.join(YAML_DIR,
                                              f"{exp_string}_rs_report",
                                              f'rs_report_{representation_id}_{epoch_num}.yml')

            print(f"ClayRS model for {epoch_num} epochs saved into {fname_cbrs}!")
            print(f"Report of the RecSys phase for {epoch_num} epochs saved into {output_report_path}!")

            # if this is the last epoch we do not print the epoch separator ("-")
            if epoch_num != ExperimentConfig.epochs[-1]:
                print("".center(80, '-'))

        # if this is the last representation to use we do not print the representation separator ("*")
        if representation_id != field_representation_list[-1]:
            print("".center(80, "*"))
