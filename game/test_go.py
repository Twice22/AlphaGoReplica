import unittest
import numpy as np
import pachi_py

from go import GoGame

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

	def make_random_board(self, size):
		game = GoGame(player_color='black', board_size=9)
		c = pachi_py.BLACK
		for _ in range(50):
			game.board = game.board.play(np.random.choice(game.board.get_legal_coords(c)), c)
			c = pachi_py.stone_other(c)

		return game

	def test_board(self):
		game = self.make_random_board(9)
		assert all(game.board[ij[0], ij[1]] == pachi_py.BLACK for ij in game.board.black_stones)
		assert all(game.board[ij[0], ij[1]] == pachi_py.WHITE for ij in game.board.white_stones)
		game.render()
		print(game.get_winner())



if __name__ == '__main__':
	unittest.main()