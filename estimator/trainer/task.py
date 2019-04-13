import os
import argparse
import tensorflow as tf

import train
from utils import create_configuration


def initialize_hyper_params(arg_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable
    Args:
        args_parser
    """


    ###########################################
    #     Experiment arguments - training     #
    ###########################################
    args_parser.add_argument(
        '--train-batch-size',
        help="Batch size for training steps",
        type=int,
        default=64
    )
    args_parser.add_argument(
        '--n-epochs',
        help="Number of epochs during training",
        type=int,
        default=10000
    )
    args_parser.add_argument(
        '--steps-to-train',
        help="Number of training steps to take, If not set \
             iterates once over the training data",
        type=int,
        default=None
    )
    args_parser.add_argument(
        '--shuffle-buffer-size',
        help="Size of buffer used to shuffle train examples",
        type=int,
        default=2000
    )
    args_parser.add_argument(
        '--export-path',
        help='Where to export the model after training',
        type=str,
        default=None
    )

    ###########################################
    #    Experiment arguments - evaluation    #
    ###########################################
    args_parser.add_argument(
        '--n-eval-games',
        help="Number of games to play during evaluation",
        type=int,
        default=100  # 400 in the paper
    )

    args_parser.add_argument(
        '--n-games',
        help="Number of self-play games played at each iteration to generate data",
        type=int,
        default=2  # 25000 in the paper
    )

    ###########################################
    #             MCTS parameters             #
    ###########################################

    args_parser.add_argument(
        '--n-mcts-sim',
        help="Number of MCTS simulations per game",
        type=int,
        default=200  # 1600 in the paper
    )
    args_parser.add_argument(
        '--c-puct',
        help="Constant to balance between exploration and exploitation",
        type=float,
        default=4.0  # Not given in the paper
    )
    args_parser.add_argument(
        '--temperature',
        help="The different temperature values",
        nargs='+',
        default=[1, 0.1],  # 1 and -> 0 in the paper
        type=float
    )
    args_parser.add_argument(
        '--temperature-scheduler',
        help="Step at which to decay the temperature parameter",
        type=int,
        default=30  # 30 first step = 1 then -> 0 afterwards in the paper
    )
    args_parser.add_argument(
        '--eta',
        help="Eta for the Dirichlet noise added to the prior",
        type=float,
        default=0.03  # Same in the paper
    )
    args_parser.add_argument(
        '--epsilon',
        help="Epsilon value use to weight the amount of Dirichlet noise to add to the prior",
        type=float,
        default=0.25  # Same in the paper
    )

    ###########################################
    #           Estimator arguments           #
    ###########################################

    # See page 18 of the paper
    args_parser.add_argument(
        '--learning-rates',
        help="Learning rate value for the optimizer",
        nargs='+',
        default=[0.01, 0.001, 0.0001], # Same in the paper
        type=float
    )
    args_parser.add_argument(
        '--learning_rates-scheduler',
        help="Number of step at which to decay the learning rate",
        nargs='+',
        default=[400000, 600000] # Same in the paper
    )
    args_parser.add_argument(
        '--momentum-rate',
        help="Momentum rate value for the optimizer",
        default=0.9, # Same in the paper
        type=float
    )
    args_parser.add_argument(
        '--n-res-blocks',
        help="Number of blocks for the residual tower",
        default=5 # 19 or 39 in the paper
    )
    args_parser.add_argument(
        '--l2_regularization',
        help="L2 regularization parameter for the weights",
        default=1e-4 # Same in the paper
    )
    args_parser.add_argument(
        '--pol-conv-width',
        help="Number of filters for the policy head",
        default=2 # 2 in the paper
    )
    args_parser.add_argument(
        '--val-conv-width',
        help="Number of filters for the value head",
        default=1 # 1 in the paper
    )
    args_parser.add_argument(
        '--conv-width',
        help="Number of filters for the convolutional layers",
        default=256 # 256 in the paper
    )
    args_parser.add_argument(
        '--fc_width',
        help="Number of units for the dense layer",
        default=256 # 256 in the paper
    )
    args_parser.add_argument(
        '--summary-steps',
        help="Number of steps between logging summary scalars",
        type=int,
        default=256
    )
    args_parser.add_argument(
        '--keep_checkpoint_max',
        help="Number of checkpoints to keep",
        type=int,
        default=5
    )
    args_parser.add_argument(
        '--mean-square-weight',
        help="weight to trade-off the value cost (mean_squared_error between z and v)",
        type=int,
        default=1
    )

    ###########################################
    #              Game parameter             #
    ###########################################

    args_parser.add_argument(
        '--n-rows',
        help="Size of the Go board",
        default=9,  # 19 (real size of the Go board) in the paper
        type=int
    )
    args_parser.add_argument(
        '--n-cols',
        help="Size of the Go board",
        default=9,  # 19 (real size of the Go board) in the paper
        type=int
    )
    args_parser.add_argument(
        '--history',
        help="Number of last states to keep for non fully observable games",
        default=8,  # Same in the paper
        type=int
    )
    args_parser.add_argument(
        '--win-ratio',
        help="Number of wins between the new and the old model to replace the old model by the new",
        default=0.55,  # Same in the paper
        type=float
    )
    args_parser.add_argument(
        '--use-random-symmetry',
        help="Whether or not to use random symmetries during inference",
        default=True,
        type=bool
    )
    args_parser.add_argument(
        '--is-go',
        help="Whether or not we are playing go",
        default=True,
        type=bool
    )

    ###########################################
    #          Saved model arguments          #
    ###########################################

    args_parser.add_argument(
        '--job-dir',
        help="GCS location to write the checkpoints, logs and the export models",
        # required=True
        default="model"
    )
    args_parser.add_argument(
        '--keep-checkpoint-max',
        help="Maximum checkpoints to keep on the disk",
        default=5,
        type=int
    )
    args_parser.add_argument(
        '--summary-step',
        help="Save summaries every this many steps",
        default=256,
        type=int
    )
    args_parser.add_argument(
        '--main-data-dir',
        help='parent directory of all the data directories',
        type=str,
        default="data"
    )
    args_parser.add_argument(
        '--selfplay-dir',
        help='Path where the data of the seflplay games are saved',
        type=str,
        default="selfplay_data"
    )
    args_parser.add_argument(
        '--holdout-dir',
        help='Path where the data of the heldout games are saved',
        type=str,
        default="heldout_data"
    )
    args_parser.add_argument(
        '--holdout-pct',
        help='ratio of games data to use for the validation set',
        type=float,
        default=0.05
    )
    args_parser.add_argument(
        '--sgf-dir',
        help='Where to write SGF (Standard Game Format) files',
        type=str,
        default=None
    )


    ###########################################
    #             Logging arguments           #
    ###########################################
    # TODO: add logging everywhere in the program
    args_parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    return args_parser.parse_args()


# python task.py --job-dir "model"
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    HYPER_PARAMS = initialize_hyper_params(args_parser)
    create_configuration(HYPER_PARAMS.__dict__)
    train.start("temp")
