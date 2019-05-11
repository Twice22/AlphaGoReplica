import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

###########################################
#     Experiment arguments - training     #
###########################################
flags.DEFINE_integer("train_batch_size", 64, "Batch size for training steps")
flags.DEFINE_integer("n_epochs", 10000, "Number of epochs during training")
flags.DEFINE_integer(
	"steps_to_train", None,
	"Number of training steps to take. If not set iterates once over the training data")
flags.DEFINE_integer(
	"shuffle_buffer_size", 2000,
	"Size of buffer used to shuffle train examples")
flags.DEFINE_string("export_path", None, "Where to export the model after training")

###########################################
#    Experiment arguments - evaluation    #
###########################################
flags.DEFINE_integer(
	"n_eval_games", 2,  # 400 in the paper (TODO: put 100 back for debug)
	"Number of games to play during evaluation")
flags.DEFINE_integer(
	"n_games", 2,  # 25000 in the paper
	"NNumber of self-play games played at each iteration to generate data")

###########################################
#             MCTS parameters             #
###########################################
flags.DEFINE_integer(
	"n_mcts_sim", 200,  # 1600 in the paper
	"Number of MCTS simulations per game")
flags.DEFINE_float(
	"c_puct", 4.0,  # Not given in the paper
	"Constant to balance between exploration and exploitation")
flags.DEFINE_multi_float(
	"temperature", [1, 0.1],  # 1 then (limit) --> 0 in the paper
	"The different temperature values")
flags.DEFINE_integer(
	"temperature_scheduler",
	30,  # 30 first step = 1 then -> 0 afterwards in the paper
	"Step at which step to decay the temperature parameter")
flags.DEFINE_float(
	"eta", 0.03,  # 0.03 in the paper
	"Eta for the Dirichlet noise added to the prior")
flags.DEFINE_float(
	"epsilon", 0.25,  # 0.25 in the paper
	"Epsilon value use to weight the amount of Dirichlet noise to add to the prior")

###########################################
#           Estimator arguments           #
###########################################
flags.DEFINE_multi_float(
	"learning_rates", [0.01, 0.001, 0.0001],  # [0.01, 0.001, 0.0001] in the paper
	"Learning rate value for the optimizer")
flags.DEFINE_multi_integer(
	"learning_rates_scheduler", [400000, 600000],  # [400000, 600000] in the paper
	"Number of step at which to decay the learning rate")
flags.DEFINE_float(
	"momentum_rate", 0.9,  # 0.9 in the paper
	"Momentum rate value for the optimizer")
flags.DEFINE_integer(
	"n_res_blocks", 5,  # 19 or 39 in the paper
	"Number of blocks for the residual tower")
flags.DEFINE_float(
	'l2_regularization', 1e-4,  # 1e-4 in the paper
	"L2 regularization parameter for the weights")
flags.DEFINE_integer(
	"pol_conv_width", 2,  # 2 in the paper
	"Number of filters for the policy head")
flags.DEFINE_integer(
	"val_conv_width", 1,  # 1 in the paper
	"Number of filters for the value head")
flags.DEFINE_integer(
	"conv_width", 256,  # 256 in the paper
	"Number of filters for the convolutional layers")
flags.DEFINE_integer(
	"fc_width", 256,  # 256 in the paper
	"Number of units for the dense layer")
flags.DEFINE_integer(
	"summary_steps", 256,
	"Number of steps between logging summary scalars")
flags.DEFINE_integer(
	"keep_checkpoint_max", 5,
	"Number of checkpoints to keep")
flags.DEFINE_float(
	'mean_square_weight', 1.0,  # 1 in the paper
	"weight to trade-off the value cost (mean_squared_error between z and v)")

###########################################
#              Game parameter             #
###########################################
flags.DEFINE_integer(
	"n_rows", 9,  # 19 (real size of the Go board) in the paper
	"Size of the Go board")
flags.DEFINE_integer(
	"n_cols", 9,  # 19 (real size of the Go board) in the paper
	"Size of the Go board")
flags.DEFINE_integer(
	"history", 8,  # 8 in the paper
	"Number of last states for each player to keep for non fully observable games")
flags.DEFINE_float(
	'win_ratio', 0.0,  # 0.55 in the paper (TODO: put back 0.55)
	"Number of wins between the new and the old model to replace the old model by the new")
flags.DEFINE_bool(
	"use_random_symmetry", True,
	"Whether or not to use random symmetries during inference")
flags.DEFINE_bool(
	"is_go", True,  # TODO: implement tic tac toe game and and use this variable
	"Whether or not we are playing go")

###########################################
#          Saved model arguments          #
###########################################
flags.DEFINE_string(
	"job_dir", "model",
	"GCS location to write the checkpoints, logs and the export models")
flags.DEFINE_string(
	"main_data_dir", "data",
	"parent directory of all the data directories")
flags.DEFINE_string(
	"selfplay_dir", "selfplay_data",
	"Path where the data of the seflplay games are saved")
flags.DEFINE_string(
	"holdout_dir", "heldout_data",
	"Path where the data of the heldout games are saved")
flags.DEFINE_float(
	'holdout_pct', 0.05,
	"ratio of games data to use for the validation set")
flags.DEFINE_string(
	"sgf_dir", None,
	"Where to write SGF (Standard Game Format) files")


if __name__ == '__main__':
	tf.app.run()