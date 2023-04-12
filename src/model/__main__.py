from src import ExperimentConfig

def main():

    if ExperimentConfig.experiment == "comparison":
        print(" Performing Cornac experiment ".center(80, '#'))
        print()
        from src.model.cornac_experiment import main as cornac_main  # perform import here just for pretty printing
        cornac_main()
        print()
        print()

        print(" Performing ClayRS experiment ".center(80, '#'))
        print()
        from src.model.clayrs_experiment import main as clayrs_main  # perform import here just for pretty printing
        clayrs_main()
        print()
        print()
    else:
        print(" Performing ClayRS additional experiment with feature extraction ".center(80, '#'))
        print()
        from src.model.additional_experiment import main as additional_main  # perform import here just for pretty printing
        additional_main()
        print()
        print()


if __name__ == "__main__":
    main()
