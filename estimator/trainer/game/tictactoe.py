import numpy as np
import six
import sys

from copy import deepcopy


def _is_terminal(mask, board_size):
    """
    Args:
        mask (nd.array): nd.array of boolean of size (board_size, board_size)
            containing True in the position of the current player
        board_size (int): size of the board

    Returns:
        Whether the game ended or not

    """
    return (board_size in mask.sum(axis=1)) | np.diag(mask).all()


class IllegalMove(Exception):
    def __init__(self, y, x):
        self.message = "Illegal Move: The spot (%i, %i) is already taken" % (y, x)
        super().__init__(self.message)


class TicTacToeGame():
    def __init__(self, player_color='black', board_size=3):
        self.board_size = board_size
        self.state = np.zeros((board_size, board_size), dtype=np.uint8)
        self.done = False  # Whether the game ended or not
        self.score = 0  # self.score = 1 if white wins, -1 if black wins

        colormap = {
            'black': 1,
            'white': 2,
        }

        self.player_color = colormap[player_color]

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _act(self, action):
        # update board  (self.state)
        y, x = action // self.board_size, action % self.board_size
        if self.state[y, x] != 0:
            raise IllegalMove(y, x)

        self.state[y, x] = self.player_color

        # update self.done
        cur_player_map = self.state == self.player_color
        if np.sum(cur_player_map) >= self.board_size:
            # check all row and one diagonal
            done1 = _is_terminal(cur_player_map)

            # rotate 90° and check row and diag again
            done2 = _is_terminal(np.rot90(cur_player_map))

            self.done = done1 | done2
            if self.done:
                self.score = 1 if self.player_color == 2 else -1

            # the game can be done if there is no space left
            self.done = self.done | (self.state != 0).all()

        # switch player (self.player_color)
        self.player_color ^= 3

    def play_action(self, action):
        if not self.done:
            try:
                # This call update self.state and self.done
                self._act(action)
            except IllegalMove:
                six.reraise(*sys.exc_info())

        return self.state, self.done

    def reset(self):
        self.done = False
        self.score = 0
        self.state = np.zeros((self.board_size, self.board_size), dtype=np.uint8)

        return self.state

    def get_legal_actions(self):
        """
        Example:
            if the board is:
            [[0, 0, 1],
           [0, 1, 0],
           [0, 2, 2]]
           Then it returns: [0, 1, 3, 5, 6]

        Returns:
            nd.array(int): flatten coords of legal actions

        """
        y, x = np.where(self.state == 0)
        return self.board_size * y + x

    def get_reward(self):
        # The winner is the player of the previous color
        white_wins = self.score > 0
        black_wins = self.score < 0

        # TODO: isn't it maybe white_wins and self.player_color = BLACK (because we have changed the color
        #  just after white had played?)
        player_wins = (white_wins and self.player_color == 1) \
                      or (black_wins and self.player_color == 2)

        reward = 1 if player_wins else -1 if (white_wins ^ black_wins) else 0

        return reward

    def get_result_string(self):
        players = {
            True: 'W',
            False: 'B'
        }
        winner = players[self.score > 0]

        if self.score == 0:
            return winner + "+" + players[not(self.score > 0)]

        return winner + '+' + str(abs(self.score))

    def get_states(self):
        return self.state.copy()


    ## Add render fct
