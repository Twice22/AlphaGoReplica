# Self-play script (see p.8 under Self-play of the paper)
import os
import time
import tensorflow as tf
import numpy as np

import trainer.config as config
import trainer.model as model
import trainer.records as records
import trainer.mcts as mcts
import trainer.game.go as go


def play(model):
    """Plays a self-play match

    Args:
        model (NeuralNetwork): current Neural Network
    Returns:
        player (MCTS): MCTS instance that records `pis`, `v` and `state` at
            each step of the game

    """
    # Notes
    #  - don't use a resign_threshold yet
    #  - don't use parallel task yet
    player = mcts.MCTS(model)

    value = 0
    count = 0

    game = go.GoGame()
    node = mcts.TreeNode()

    while True:
        # by default: first 30 moves of the game, the temperature is set to 1
        # TODO: debug node? Is node update at each iteration?
        if count < config.temperature_scheduler:
            probs, action = player.search(game, node, config.temperature[0])
        else:
            probs, action = player.search(game, node, config.temperature[1])

        # self.root.position.n represents the # of moves played so far
        # self.search_pi is update each time play_move is used on player: it adds
        # pi vector at each move
        # self.result is only retrieved at the end of the while True loop and is
        # set before breaking off the loop so that player.result = -1 or 1
        # go.replay_position uses self.position.recent -> Tuple of player moves (color, move)
        # in extract_data: pwc.position is a Position object new for each position
        # of the game. pwc.result is the int (-1 or 1) received at the end of the game
        # pi is the probs at each step of the game

        # once the data is extracted, we use preprocessing.make_dataset_from_selfplay(extracted_data):
        # for pos, pi, result in data_extracts... as pos is a Postion object we need to preprocess it
        # before passing it to make_tf_examples (-> create_example for me) because pos need to
        # be a (n_rows, n_cols, 2 * history + 1) tensor. Just need to use  self.state from the game object

        # by doing that we ensure we have n x 362 tensor of floats representing the
        # mcts search probabilities where n is the number of moves in the game

        _, game_over = game.play_action(action)

        count += 1

        if game_over:
            player.set_result(game.get_reward())
            player.set_result_string(game.get_result_string())
            break

        # The child node becomes the new root node (already
        # handled in mcts script)

    return player


# SGF format: https://fr.wikipedia.org/wiki/Smart_Game_Format
def run_game(load_file, selfplay_dir=None, holdout_dir=None,
             sgf_dir=None, holdout_pct=0.05):
    """
    Play a game and record results and game data

    Args:
        load_file (ckpt): file that contains the weights of the model
        selfplay_dir (str): directory where to write game data
        holdout_dir (str): directory where to write held-out game data
        sgf_dir (str): directory where to write SGFs
        holdout_pct (float): percentage of game to hold out
    """
    if sgf_dir is not None:
        full_sgf_dir = os.path.join(config.main_data_dir, sgf_dir)
        tf.gfile.MakeDirs(full_sgf_dir)
    if selfplay_dir is not None:
        full_selfplay_dir = os.path.join(config.main_data_dir, selfplay_dir)
        full_holdout_dir = os.path.join(config.main_data_dir, holdout_dir)
        tf.gfile.MakeDirs(full_selfplay_dir)
        tf.gfile.MakeDirs(full_holdout_dir)

    # initialize the NN with the weights `load_file`
    network = model.NeuralNetwork(load_file)

    # selfplay game until the end using mcts algorithm
    player = play(network)

    game_data = player.extract_data()
    output_name ="%i_game" % int(time.time())

    # save the data in SGF format
    if sgf_dir is not None:
        with tf.gfile.GFile(sgf_dir, '%s.sgf' % output_name, "w") as f:
            f.write(player.to_sgf())

    tf_examples = records.make_selfplay_dataset(game_data)

    if selfplay_dir is not None:
        # Hold out 5% of all the games for validation
        if np.random.random() < holdout_pct:
            file_name = os.path.join(holdout_dir,
                                     "%s.tfrecord" % output_name)
        else:
            file_name = os.path.join(selfplay_dir,
                                     "%s.tfrecord" % output_name)

        # save the data into tfrecords. One folder contains
        # data to validate the performance of the model and
        # the other contains the selfplay data
        records.write_tf_examples(file_name, tf_examples)


def main(unused_args):
    run_game(
        load_file=config.weight_folder,
        selfplay_dir=config.selfplay_dir,
        holdout_dir=config.holdout_dir,
        sgf_dir=config.sgf_dir,
        holdout_pct=config.holdout_pct,
    )


if __name__ == '__main__':
    tf.app.run(main)
