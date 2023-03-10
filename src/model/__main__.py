from src.model.clayrs_experiment import main as clayrs_main


def main():
    print(" Performing Cornac experiment ".center(80, '#'))
    print()
    from src.model.cornac_experiment import main as cornac_main  # perform import here just for pretty printing
    cornac_main()
    print()
    print()

    print(" Performing ClayRS experiment ".center(80, '#'))
    print()
    clayrs_main()
    print()
    print()


if __name__ == "__main__":
    main()
