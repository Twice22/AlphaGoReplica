import numpy as np
import sgfwriter

from config import *
from copy import deepcopy
from utils import get_komi


def dirichlet_noise(P_s):
    # see Self-play p.8 of the paper
    # add dirichlet noise to the prior probabilities of the root
    # node to allow for additional exploration
    size = len(P_s)
    P_sa = (1 - FLAGS.epsilon) * P_s + FLAGS.epsilon * np.random.dirichlet(np.full(size, FLAGS.eta))
    return P_sa


# cannot be part of Node class because we need \sum_b N(s, b) i.e 
# how many times we visited state s
# see Select (Fig.2a) p.8 of the paper 
def select_action(nodes):
    """ nodes should have Q(s, a), P(s, a), N(s, a) action_id for all a  """
    PUCT_sa = np.zeros(len(nodes[:, 0]))
    N_s = sum(nodes[:, 2])
    for i, (Q_sa, P_sa, N_sa) in enumerate(nodes):
        PUCT_sa[i] = Q_sa + FLAGS.c_puct * P_sa * (N_s ** 0.5) / (1 + N_sa)

    # if 2 or more actions have the same confidence bound return one of them at random
    best_idx = np.where(PUCT_sa == np.max(PUCT_sa))[0]
    return np.random.choice(best_idx)


class Node:
    def __init__(self, parent=None, prob=0, action=None):
        # initialization (see 'Expand and evaluate' p.8 of the paper)
        self.Q_sa = 0  # mean value of the state, action pair (s, a)
        self.N_sa = 0  # number of times the pair (s,a) as been visited
        self.W_sa = 0  # total action value
        self.P_sa = prob  # prior probability of selecting that edge: P(a|s)
        self.parent = parent
        self.action = action
        self.children = []  # list of Node

    def is_leaf(self):
        """ Returns whether the node is a leaf or not """
        return not self.children

    # see 'Backup' p.8 of the paper
    def back_prop(self, v):
        """
        Args:
            v: value of the node at the end of MCTS
        Update the edges statistics W_sa and Q_sa
        """
        self.W_sa += v
        self.N_sa += 1
        self.Q_sa = self.W_sa / self.N_sa

    def expand_nodes(self, P_s):
        """
        Args:
            P_s: probability of selecting each action
            Populate the children variable with action whose probability is non-zero
        """
        self.children = [Node(self, prob, action) for action, prob in enumerate(P_s) if prob > 0]


class MCTS:
    def __init__(self, net):
        self.net = net
        self.root = None
        self.game = None

        # records the probabilities at each step
        self.probs_history = []

        # record the state of the game at each step
        self.states_history = []

        # record the actions (moves) at each step
        self.actions_history = []

        # result of the game (Default to 0 -> tie)
        # at the end of the game result should be 1 or -1 (tie not allowed)
        self.result = 0
        self.result_string = None

    # Play (Fig.2d) page 8
    # select the move to expand according to how many times we visited all other nodes
    def select_move(self, temp):
        rev_temp = 1 / temp
        probs = [node.N_sa ** rev_temp for node in self.root.children]
        probs /= np.sum(probs)

        # return idx of self.root.children
        idx = np.random.choice(len(probs), p=probs)

        # ensure to return a vector of size (n_rows * n_cols + 1)
        p = np.zeros(FLAGS.n_rows * FLAGS.n_cols + 1)
        p[np.array([node.action for node in self.root.children])] = probs

        return p, idx

    def set_result(self, val):
        self.result = val

    def set_result_string(self, val):
        self.result_string = val

    def extract_data(self):
        assert len(self.probs_history) == len(self.states_history)
        for state, pi in zip(self.states_history, self.probs_history):
            yield state.astype(np.int8), pi.astype(np.float32), self.result

    def to_sgf(self):
        assert self.result_string is not None

        return sgfwriter.write_sgf(self.actions_history,
                                   self.result_string,
                                   komi=get_komi())

    def search(self, game, node, temp):
        self.root = node
        self.game = game

        # do `n_mcts_sim` of Monte Carlo
        for i in range(FLAGS.n_mcts_sim):
            node = self.root
            done = False
            game = deepcopy(self.game)  # use override deepcopy version from the goGame class

            # loop till node have children (till game not finished)
            while not node.is_leaf() and not done:
                Q_P_N = np.array([[child_node.Q_sa, child_node.P_sa, child_node.N_sa] for child_node in node.children])
                best_idx = select_action(Q_P_N)
                node = node.children[best_idx]

                # use that action in the game
                state, done = game.play_action(node.action)

            # get probabilities P_s and values from the network for this state
            P_s, v = self.net.run(game)
            P_s, v = P_s[0], v[0]

            # Add Dirichlet noise to the root node
            # because we initialize the node with `node = mcts.TreeNode()`
            # in `self_play.py` and  by default TreeNode takes parent=None
            if node.parent is None:
                P_s = dirichlet_noise(P_s)

            # get available actions
            legal_actions = game.get_legal_actions()

            # get illegal actions: all actions - available actions
            illegal_actions = np.setdiff1d(np.arange(game.board_size ** 2 + 1), np.array(legal_actions))

            # re-normalize the P_s vector P_s = [P(s, a1), P(s, a2), ..., P(s, aN)]
            # with P(s, ai) scalar
            P_s[illegal_actions] = 0
            sum_P = np.sum(P_s)
            if np.sum(P_s) > 0:
                P_s /= sum_P

            # add children of the new node we are in if possible
            node.expand_nodes(P_s)

            # if game is over, override the value `v` returned by the neural network
            # with the real value (+1 or -1)
            if done:
                v = game.get_reward()

            # Backpropagate node statistics from leaf to root node
            while node is not None:
                node.back_prop(v)  # v is retrieve from evaluating the game using the CNN
                node = node.parent

        # see Play (Fig. 2d) p.8 of the paper
        # AlphaGo Zero selects a move a to play in the root position
        # proportional to its exponentiated visit count
        probs, idx = self.select_move(temp)
        action = self.root.children[idx].action

        # add probability vector pi to probs_history
        # each time we chose a move from using the mcts search
        self.probs_history.append(probs)

        # append state to states_history at each step of the game
        self.states_history.append(game.get_states())

        # record the action at each step
        self.actions_history.append(action)

        # probs = vector of probs of size [n_rows * n_cols + 1, ]
        # action = scalar in range [1, n_rows * n_cols + 1]
        # self.root.children[idx] = new root node
        return probs, action, self.root.children[idx]


