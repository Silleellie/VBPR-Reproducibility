"""
Main module of the model training phase
"""
from src import ExperimentConfig

def main():
    """
    Main which performs the model training phase by calling functions w.r.t. the operations to carry out for the
    specified experiment type (comparison or additional)
    """

    if ExperimentConfig.experiment == "comparison":
        print(" Performing Cornac experiment ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.cornac_experiment import main as cornac_main
        cornac_main()
        print()
        print()

        print(" Performing ClayRS experiment ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.clayrs_experiment import main as clayrs_main
        clayrs_main()
        print()
        print()
    else:
        print(" Performing ClayRS additional experiment with feature extraction ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.model.additional_experiment import main as additional_main
        additional_main()
        print()
        print()


if __name__ == "__main__":
    main()
