import unittest
import numpy as np
import pachi_py

from go import GoGame


def make_random_board(size):
	game = GoGame(player_color='black', board_size=9)
	c = pachi_py.BLACK
	for _ in range(50):
		game.board = game.board.play(np.random.choice(game.board.get_legal_coords(c)), c)
		c = pachi_py.stone_other(c)

	return game

class TestGoGame(unittest.TestCase):
	""" Class that contains unit tests for the GoGame class """

	def test_change_player(self):
		game = GoGame(player_color='black', board_size=9)
		self.assertEqual(game.player_color, 1) # black
		game.play_action(15)
		self.assertEqual(game.player_color, 2) # white

		game = GoGame(player_color='white', board_size=9)
		self.assertEqual(game.player_color, 2) # white
		game.play_action(7)
		self.assertEqual(game.player_color, 1) # black

	def test_reset(self):
		board_size = 9
		game = GoGame(player_color='black', board_size=board_size)
		game.reset()

		self.assertEqual(game.done, False)
		np.testing.assert_array_equal(game.state, np.zeros((17, board_size, board_size)))

	def test_illegal_move(self):
		game = GoGame(player_color='black', board_size=9)
		game.play_action(14)
		try:
			game.play_action(14)
		except pachi_py.IllegalMove:
			return
		assert False, "IllegalMove exception should have been raised"

	def test_suicide_move(self):
		game = GoGame(player_color='black', board_size=9)
		for action in [22, 1, 30, 2, 32, 3, 40]:
			game.play_action(action)
		try:
			game.play_action(31)
		except pachi_py.IllegalMove:
			return
		assert False, "SuicideMove exception should have been raised"

	def test_board(self):
		game = make_random_board(9)
		assert all(game.board[ij[0], ij[1]] == pachi_py.BLACK for ij in game.board.black_stones)
		assert all(game.board[ij[0], ij[1]] == pachi_py.WHITE for ij in game.board.white_stones)


	def test_score(self):
		# actions = [40, 38, 42, 21, 47, 46, 56, 39, 49, 55, 22, 13, 23,
		# 		   48, 57, 58, 67, 59, 68, 60, 64, 43, 34, 51, 41, 63,
		# 		   69, 70, 79, 61, 53, 65, 66, 73, 52, 75, 50, 76, 71,
		# 		   12, 14, 5, 6, 4, 15, 31, 32, 30, 62, 77, 78, 74, 81, 81]
		# actions = [32, 57, 60, 38, 22, 20, 67, 66, 58, 49, 50, 40, 41, 12, 13,
		# 		   4, 5, 3, 14, 76, 77, 75, 68, 21, 31, 30, 81, 81]
		actions = [32, 57, 60, 24, 33, 23, 22, 13, 12, 21, 31, 11, 14, 3, 15, 39,
				   49, 48, 67, 66, 58, 76, 77, 75, 68, 5, 25, 40, 41, 6, 7, 4, 30,
				   29, 81, 81] # pass twice to end the game

		# rules of Go: black makes the first move
		game = GoGame(player_color='black', board_size=9)
		for action in actions:
			game.play_action(action)
		
		self.assertEqual(game.done, True)
		self.assertEqual(game.get_winner(), -1) # current player (black) lose

	def test_state(self):
		actions = [40, 38, 42, 21, 47, 46, 56, 39, 49, 55, 22, 13, 23,
				   48, 57, 58, 67, 59, 68, 60, 64, 43, 34, 51, 41, 63,
				   69, 70, 79, 61, 53, 65, 66, 73, 52, 75, 50, 76, 71,
				   12, 14, 5, 6, 4, 15, 31, 32, 30, 62, 77, 78, 74]
		board_size = 9

		game = GoGame(player_color='black', board_size=board_size)
		color = 0
		list_states = []
		for action in actions:
			game.play_action(action)
			list_states.append(np.array(game.board.encode()[color]))
			color = 1 - color

		state = list(game.state)[:-1]
		while state:
			s_from_state = state.pop(0)
			s_from_ground_truth = list_states.pop()
			np.testing.assert_array_equal(s_from_state, s_from_ground_truth)

		# test that current player (which is the next player to play) is the value of the last row of game.state
		np.testing.assert_array_equal(game.state[-1] - (game.player_color - 1), np.zeros((board_size, board_size)) )

if __name__ == '__main__':
	unittest.main()