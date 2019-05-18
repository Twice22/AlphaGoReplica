import pachi_py
import sgfwriter
import tensorflow as tf
from tqdm import tqdm
import os

import model
import mcts
import game.go as go

from config import *
from utils import get_komi


def evaluate(prev_model, cur_model, epoch=0):
    """Plays matches between 2 neural networks. 1 Neural network
        loads the weights from the previous model and the other loads
        the newly created weights. Evaluate which model is the best a keep it

    Args:
        prev_model (str): Path to the previous model (ckpt file)
        cur_model (str): Path to the new model (ckpt file)
        epoch (int): epoch at which we are evaluating the model
    Returns:

    """
    # Note: black plays first in the game of GO, so, in the case where the komi is not
    # set then black will have an advantage. In this eventuality we want our previous
    # model to play the black stones and our new model to use the white stones, so that
    # if white wins over black we have certain margin, it definitively means that the new
    # model is better than the previous model
    black_model = model.NeuralNetwork(prev_model)
    white_model = model.NeuralNetwork(cur_model)

    black = mcts.MCTS(black_model)
    white = mcts.MCTS(white_model)

    # create sgf_dir if it doesn't exist
    if FLAGS.sgf_dir is not None:
        full_sgf_dir = os.path.join(FLAGS.main_data_dir, FLAGS.sgf_dir)
        tf.gfile.MakeDirs(full_sgf_dir)

    wins = 0  # number of wins of the new model (white player) over the previous model

    for i in tqdm(range(FLAGS.n_eval_games), ncols=100, desc='\tGame evaluation'):
        node = mcts.Node()
        game = go.GoGame()  # create new game for each evaluation
        value = 0  # By default value = 0 (tie)
        actions_history = []

        while True:
            # If player_color == 1 (BLACK) use mcts with weights from the black model
            # else (player_color == 2), use mcts with weights from the white model
            if game.player_color == 1:
                probs, action, best_child = black.search(game, node, FLAGS.temperature[1])
            else:
                probs, action, best_child = white.search(game, node, FLAGS.temperature[1])

            _, game_over = game.play_action(action)
            actions_history.append(action)

            # make the child node the new root node
            best_child.parent = None
            node = best_child

            if game_over:
                reward = game.get_reward()
                assert reward != 0  # TODO: Ensure reward should be different from 0

                # save the game as sgf
                filename = "epoch{:d}-eval{:d}-W-vs-B.sgf".format(epoch, i)
                with tf.gfile.GFile(os.path.join(full_sgf_dir, filename), 'w') as f:
                    sgf_res = sgfwriter.write_sgf(actions_history, game.get_result_string(), komi=get_komi())
                    f.write(sgf_res)

                # value = 1 if white (new model) player wins, 0 otherwise
                value = int(game.player_color == pachi_py.WHITE and reward > 0)
                break

        wins += value

    return wins / FLAGS.n_eval_games


def main(argv):
    """Play matches between two neural networks"""
    _, prev_model, cur_model = argv
    evaluate(prev_model, cur_model)


if __name__ == '__main__':
    main()
