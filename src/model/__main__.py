"""
Main module of the model training phase
"""

from src import ExperimentConfig


def main():
    """
    Main which performs the model training phase by calling functions w.r.t. the operations to carry out for the
    specified experiment type (comparison or additional)

    """

    if ExperimentConfig.experiment == "exp1":
        print(" Performing Cornac experiment ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.exp1_cornac_experiment import main as cornac_main_exp1
        cornac_main_exp1()
        print()
        print()

        print(" Performing ClayRS experiment ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.exp1_clayrs_experiment import main as clayrs_main_exp1
        clayrs_main_exp1()
        print()
        print()

    elif ExperimentConfig.experiment == "exp2":
        print(" Performing ClayRS additional experiment with feature extraction ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.exp2_caffe import main as caffe_main_exp2
        caffe_main_exp2()
        print()
        print()

    else:
        print(" Performing ClayRS additional experiment with feature extraction ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.exp3_vgg19_resnet import main as vgg_resnet_main_exp3
        vgg_resnet_main_exp3()
        print()
        print()


if __name__ == "__main__":
    main()
