import os
import argparse

import tensorflow as tf

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
        default=100
    )


    ###########################################
    #    Experiment arguments - evaluation    #
    ###########################################
    args_parser.add_argument(
        '--n-eval-games',
        help="Number of games to play during evaluation",
        type=int,
        default=100 # 400 in the paper
    )

    args_parser.add_argument(
        '--n-games',
        help="Number of self-play games played at each iteration to generate data",
        type=int,
        default=400 # 25000 in the paper
    )

    ###########################################
    #             MCTS parameters             #
    ###########################################

    args_parser.add_argument(
        '--n-mcts-sim',
        help="Number of MCTS simulations per game",
        type=int,
        default=200 # 1600 in the paper
    )
    args_parser.add_argument(
        '--c-puct',
        help="Constant to balance between exploration and exploitation",
        type=float,
        default=4.0 # Not given in the paper
    )
    args_parser.add_argument(
        '--temperature',
        help="The different temperature values",
        nargs='+',
        default=[1, 0.001], # 1 and -> 0 in the paper
        type=float
    )
    args_parser.add_argument(
        '--temperature-scheduler',
        help="Step at which to decay the temperature parameter",
        type=int,
        default=30 # 30 first step = 1 then -> 0 afterwards in the paper
    )
    args_parser.add_argument(
        '--dirichlet-alpha',
        help="Alpha for the Dirichlet noise added to the prior",
        type=float,
        default=0.03 # Same in the paper
    )
    args_parser.add_argument(
        '--epsilon',
        help="Epsilon value use to weight the amount of Dirichlet noise to add to the prior",
        type=float,
        default=0.25 # Same in the paper
    )

    ###########################################
    #           Estimator arguments           #
    ###########################################

    # See page 18 of the paper
    args_parser.add_argument(
        '--learning_rates',
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
        '--momentum_rate',
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

    ###########################################
    #              Game parameter             #
    ###########################################

    args_parser.add_argument(
        '--n-rows',
        help="Size of the Go board",
        default=9, # 19 (real size of the Go board) in the paper
        type=int
    )
    args_parser.add_argument(
        '--n-cols',
        help="Size of the Go board",
        default=9, # 19 (real size of the Go board) in the paper
        type=int
    )
    args_parser.add_argument(
        '--history',
        help="Number of last states to keep for non fully observable games",
        default=8, # Same in the paper
        type=int
    )
    args_parser.add_argument(
        '--win-ratio',
        help="Number of wins (in %) between the new and the old model to replace the old model by the new",
        default=55, # Same in the paper
        type=int
    )
    args_parser.add_argument(
        '--use-random-symmetry',
        help="Wether or not to use random symmetries during inference",
        default=True,
        type=bool
    )

    ###########################################
    #          Saved model arguments          #
    ###########################################

    args_parser.add_argument(
        '--job_dir',
        help="GCS location to write the checkpoints, logs and the export models",
        required=True
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

    ###########################################
    #               TPU arguments             #
    ###########################################
    arg_parser.add_argument(
        '--use-tpu',
        help="Wether to use TPU for training", # needed because we need to convert the code for TPU
        default=False
    )

    # TODO: add TPU arguments if we want to
    # be able to train the model using TPU.
    # need to use TPU estimators...


    ###########################################
    #             Logging arguments           #
    ###########################################
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


def run_experiment(run_config):
     """ Train, evaluate, and export the model
        using tf.estimator.train_and_evaluate API
    Args:
        run_config (tf.estimator.RunConfig): Configuration file
        that specifies the parameters for the general model such
        as where to save the checkpoints, how many maximum checkpoints
        to save, ...

     """
     pass # not implemented yet

# python task.py --job_dir "model"
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    HYPER_PARAMS = initialize_hyper_params(args_parser)
    create_configuration(HYPER_PARAMS.__dict__)