import config
import model
import mcts
import game.go as go
import pachi_py


def evaluate(prev_model, cur_model):
    """Plays matches between 2 neural networks. 1 Neural network
        loads the weights from the previous model and the other loads
        the newly created weights. Evaluate which model is the best a keep it

    Args:
        prev_model (str): Path to the previous model (ckpt file)
        cur_model (str): Path to the new model (ckpt file)
    Returns:

    """
    # Note: black plays first in the game of GO, so, in the case where the komi is not
    # set then black will have an advantage. In this eventuality we want our previous
    # model to play the black stones and our new model to use the white stones, so that
    # if white wins over black we a certain margin, it definitively means that the new
    # model is better than the previous model
    black_model = model.NeuralNetwork(prev_model)
    white_model = model.NeuralNetwork(cur_model)

    black = mcts.MCTS(black_model)
    white = mcts.MCTS(white_model)

    wins = 0  # number of wins of the new model (white player) over the previous model

    for i in range(config.n_eval_games):
        node = mcts.TreeNode()
        game = go.GoGame()  # create new game for each evaluation
        value = 0  # By default value = 0 (tie)

        while True:
            # If player_color == 1 (BLACK) use mcts with weights from the black model
            # else (player_color == 2), use mcts with weights from the white model
            if game.player_color == 1:
                probs, action, best_child = black.search(game, node, config.temperature[1])
            else:
                probs, action, best_child = white.search(game, node, config.temperature[1])

            _, game_over = game.play_action(action)

            # TODO: save game as sgf

            # make the child node the new root node
            best_child.parent = None
            node = best_child

            if game_over:
                reward = game.get_reward()
                assert reward != 0  # TODO: Ensure reward should be different from 0
                value = int(game.player_color == pachi_py.WHITE and reward > 0)
                break

        wins += value

    return wins / config.n_eval_games


def main(argv):
    """Play matches between two neural networks"""
    # TODO: create directory with sgf
    _, prev_model, cur_model = argv
    evaluate(prev_model, cur_model)


if __name__ == '__main__':
    main()
