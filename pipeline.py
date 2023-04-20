import argparse

from src import ExperimentConfig
from src.data.__main__ import main as data_main
from src.model.__main__ import main as model_main
from src.evaluation.__main__ import main as eval_main


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to reproduce the VBPR experiment')
    parser.add_argument('-epo', '--epochs', type=int, default=[5, 10, 20, 50], nargs='+',
                        help='Number of epochs for which the VBPR network will be trained', metavar='5')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
                        help='Batch size that will be used for the torch dataloaders during training',
                        metavar='128')
    parser.add_argument('-gd', '--gamma_dim', type=int, default=20,
                        help='Dimension of the gamma parameter of the VBPR network', metavar='20')
    parser.add_argument('-td', '--theta_dim', type=int, default=20,
                        help='Dimension of the theta parameter of the VBPR network', metavar='20')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005,
                        help='Learning rate for the VBPR network', metavar='0.005')
    parser.add_argument('-seed', '--random_seed', type=int, default=42,
                        help='random seed', metavar='42')
    parser.add_argument('-nt_ca', '--num_threads_ca', type=int, default=4,
                        help='Number of threads that will be used in ClayRS can see '
                             'during Content Analyzer serialization phase',
                        metavar='4')
    parser.add_argument('-exp', '--experiment', type=str, default='exp1',
                        help='exp1 to perform the comparison experiment between ClayRS can see and Cornac, '
                             'exp2 to perform end to end experiment using caffe via ClayRS can see, '
                             'exp3 to perform end to end experiment using vgg19 and resnet50 via Clayrs can see',
                        metavar='exp1')

    args = parser.parse_args()

    ExperimentConfig.random_state = args.random_seed
    ExperimentConfig.epochs = args.epochs
    ExperimentConfig.batch_size = args.batch_size
    ExperimentConfig.gamma_dim = args.gamma_dim
    ExperimentConfig.theta_dim = args.theta_dim
    ExperimentConfig.learning_rate = args.learning_rate
    ExperimentConfig.num_threads_ca = args.num_threads_ca

    if args.experiment in {"exp1", "exp2", "exp3"}:
        ExperimentConfig.experiment = args.experiment
    else:
        raise ValueError("Only 'exp1', 'exp2' or 'exp3' type of experiments are supported!")

    data_main()
    model_main()
    eval_main()
