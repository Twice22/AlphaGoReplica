import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import six
import pachi_py

# TODO: change for a variable using parser for HISTORY
# TODO: define komi variable: either 6.5 or 7.5
HISTORY = 7
BOARD_SIZE = 9


# The coordinate representation of Pachi (and pachi_py) is defined on a board
# with extra rows and columns on the margin of the board, so positions on the board
# are not numbers in [0, board_size**2) as one would expect. For this Go env, we instead
# use an action representation that does fall in this more natural range.

def _pass_action(board_size):
	""" Return the pass action """
    return board_size ** 2

def _resign_action(board_size):
	""" Return the resign action """
    return board_size ** 2 + 1

def _coord_to_action(board, c):
    """ Converts Pachi coordinates to actions """
    if c == pachi_py.PASS_COORD:
    	return _pass_action(board.size)

    if c == pachi_py.RESIGN_COORD:
    	return _resign_action(board.size)

    i, j = board.coord_to_ij(c)
    return i * board.size + j

def _action_to_coord(board, a):
    """ Converts actions to Pachi coordinates """
    if a == _pass_action(board.size):
    	return pachi_py.PASS_COORD

    if a == _resign_action(board.size):
    	return pachi_py.RESIGN_COORD

    return board.ij_to_coord(a // board.size, a % board.size)

def _format_state(history, player_color, board_size):
    """ Format the board to be used as the input to the NN.
        See Neural network architecture p.8 of the paper """
    state_history = np.concatenate((history[0], history[1]), axis=0) # TODO: change input features to be entrelaced
    to_play = np.full((1, board_size, board_size), player_color - 1) # BLACK = 1 in pachi, 0 in our env, same for WHITE = 2 -> 1 in our env
    final_state = np.concatenate((state_history, to_play), axis=0)
    return final_state

class GoGame():
    '''
    Go environment. Play against a fixed opponent.
    '''

    def __init__(self, player_color='black', board_size=9):
        """
        Args:
            player_color (str): Stone color for the agent. Either 'black' or 'white'
            board_size (int): board game side size
        """
        self.board_size = board_size

        colormap = {
            'black': pachi_py.BLACK, # pachi_py.BLACK = 1
            'white': pachi_py.WHITE, # pachi_py.WHITE = 2
        }
        self.player_color = colormap[player_color]

        # history: [(8, 9, 9), (8, 9, 9)]
        self.history = [np.zeros((HISTORY + 1, board_size, board_size)),
                        np.zeros((HISTORY + 1, board_size, board_size))]
        self.reset()

    # https://stackoverflow.com/questions/1500718/what-is-the-right-way-to-override-the-copy-deepcopy-operations-on-an-object-in-p
    def __deepcopy__(self):
    	""" Clone the current instance of the game """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k =="board":
                setattr(result, k, self.board.clone())
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


    def play_action(self, action):
        """ If 2 player passes the game end (rule of Go),
            `play_action` performs an action and update
            the state of the game
		Args:
			action (int): move to execute
		Attributes:
			done (bool): Whether the game ends or not
			state (np.array): Tensor of size (17, board_size, board_size) as
			described in the paper under `Neural network architecture` page 8
        """

        if not self.done:
            try:
                self._act(action)
            except pachi_py.IllegalMove:
                six.reraise(*sys.exc_info())

        self.done = self.board.is_terminal
        self.state = _format_state(self.history, self.player_color, self.board_size)

    def _komi(self):
    	""" Add komi (bonus point) to the second player
    		according to the size of the game in order to
    		balance the game
    	"""
        if 14 <= self.board_size <= 19:
            return 7.5
        elif 9 <= self.board_size <= 13:
            return 5.5
        return 0

    def reset(self):
    	""" reset the Game
    	Attributes:
    		komi (int): komi value (bonus point) to balance the game
    		board (PachiBoardPtr): board class with associate methods
    		done (bool): Wether the game ends or not
    		state (np.array): Tensor of size (17, board_size, board_size)
    		formatted as described in the paper under `Neural network architecture` 
    		page 8
    	"""
        self.komi = self._komi()
        self.board = pachi_py.CreateBoard(self.board_size) # object with method
        self.done = self.board.is_terminal
        self.state = np.zeros(((HISTORY + 1) * 2 + 1, BOARD_SIZE, BOARD_SIZE))

        return self.state

    def get_legal_actions(self):
        """ Get all the legal actions and transform their coords into integer values
        Returns:
       		np.array: array of integers that represents the legal actions.
        """

        legal_moves = self.board.get_legal_coords(self.player_color, filter_suicides=True)
        return np.array([_coord_to_action(self.board, pachi_move) for pachi_move in legal_moves])

    def _act(self, action):
        """ Executes an action for the current player 
		Args:
			action (int): action to execute
		Attributes:
			board (PachiBoardPtr): update the board instance
			player_color (int): Switch player
        """

        self.board = self.board.play(_action_to_coord(self.board, action), self.player_color)
        board = self.board.encode()
        color = self.player_color - 1

        # discard last history of current player and add current move to history of current player
        self.history[color] = np.roll(self.history[color], 1, axis=0)
        self.history[color][0] = np.array(board[color])

        # switch player
        self.player_color = pachi_py.stone_other(self.player_color)

    def get_winner(self):
        """ Get the winner using Tromp-Taylor scoring """

        # Tromp-Taylor scoring https://github.com/openai/pachi-py/blob/master/pachi_py/pachi/board.c#L1556
        # is called by: https://github.com/openai/pachi-py/blob/master/pachi_py/goutil.cpp#L81
        # which is called by the python API: https://github.com/openai/pachi-py/blob/master/pachi_py/cypachi.pyx#L52
        # Note: we don't have access to the komi variables through the pachi_py API, so we need to add it ourself
        score = self.komi + self.board.official_score
        white_wins = score > 0
        black_wins = score < 0

        player_wins = (white_wins and self.player_color == pachi_py.WHITE) \
                      or (black_wins and self.player_color == pachi_py.BLACK)

        reward = 1 if player_wins else -1 if (white_wins or black_wins) else 0

        return reward

    def render(self):
        """ Render the board in the console """
        outfile = sys.stdout
        outfile.write('To play: {}\n{}\n'.format(six.u(
                        pachi_py.color_to_str(self.player_color)),
                        self.board.__repr__().decode()))
        return outfile