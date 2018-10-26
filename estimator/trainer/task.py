import os
import argparse

import tensorflow as tf

# TODO: other imports?


def initialize_hyper_parals(arg_parser):
	"""
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable
    Args:
        args_parser
    """


    ###########################################
    # 	  Experiment arguments - training     #
    ###########################################
    args_parser.add_argument(
    	'--train-batch-size',
    	help="Batch size for training steps",
    	type=int,
    	default=256
    )


    ###########################################
    # 	        Estimator arguments           #
    ###########################################

    # See page 18 of the paper
    args_parser.add_argument(
    	'--learning_rates',
    	help="Learning rate value for the optimizer",
    	nargs='+',
    	default=[0.01 0.001 0.0001],
    	type=float
    )
    args_parser.add_argument(
    	'--learning_rates-scheduler',
    	help="Number of step at which to decay the learning rate",
    	nargs='+',
    	default=[400000, 600000]
    )
    args_parser.add_argument(
    	'--momentum_rate',
    	help="Momentum rate value for the optimizer",
    	default=0.9,
    	type=float
    )
    args_parser.add_argument(
    	'--n-res-blocks',
    	help="Number of blocks for the residual tower",
    	default=19
    )
    args_parser.add_argument(
    	'--l2_regularization'
    	help="L2 regularization parameter for the weights",
    	default=1e-4
    )
    arg_parser.add_argument(
    	'--working-dir',
    	help="Estimator working directory where to save the checkpoints, logs, ...",
    	default=None
    )
    arg_parser.add_argument(
    	'--use-tpu'
    	help="Wether to use TPU for training", # needed because we need to convert the code for TPU
    	default=False
    )
