import numpy as np

EPSILON = 0.25
ETA = 0.03
C_PUCT = 0.2
NB_SIM = 100


def dirichlet_noise(P_s):
    # see Self-play p.8 of the paper
    # add dirichlet noise to the prior probabilities of the root
    # node to allow for additional exploration
    size = len(P_s)
    P_sa = (1 - EPSILON) * P_s + EPSILON * np.random.dirichlet(np.full(size, ETA))
    return P_sa


# cannot be part of Node class because we need \sum_b N(s, b) i.e 
# how many times we visited state s
# see Select (Fig.2a) p.8 of the paper 
def select_action(nodes):
    """ nodes should be have Q(s, a), P(s,a), N(s, a) for all a  """
    PUCT_sa = np.zeros(nodes.shape[0])
    N_s = sum(nodes[2,:])
    for i, (Q_sa, P_sa, N_sa) in enumerate(nodes):
        PUCT_sa[i] = Q_sa + C_PUT * P_sa * (N_s ** 0.5) / (1 + N_sa)

    # if 2 or more actions have the same confidence bound return one of them at random
    best_actions = np.where(PUCT_sa == np.max(PUCT_sa))[0]
    return np.random.choice(best_actions)

class Node:
    def __init__(self, parent=None, prob=0, action=None):
        self.parent = parent
        self.prob = prob
        self.action = action

        # initialization (see 'Expand and evaluate' p.8 of the paper)
        self.Q_sa = 0 # mean value of the state, action pair (s,a)
        self.N_sa = 0 # number of times the pair (s,a) as been visited
        self.W_sa = 0 # total action value
        self.P_sa = prob # prior probability of selecting that edge: P(a|s)

        self.childrens = [] # list of Node

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
        self.Q = self.W_sa / self.N_sa

    def expand_nodes(self, P_s):
        """
        Args:
            P_s: probability of selecting each action
            Populate the children variable with action whose probability is non-zero
        """
        self.children = [Node(self, prob, idx) for idx, prob in enumerate(P_s) if prob > 0]


class MCTS:
    def __init__(self):
        self.game = None
        self.net = net

    def select_move(self, temp):
        rev_temp = 1 / temp
        pi_as = [(node.N_sa, node.action) ** rev_temp for node in self.root.childrens]
        probs = [v for v, a in pi_as]
        probs /= np.sum(probs)
        
        idx = np.random.choice(len(probs), p=probs)
        action = pi_as[idx][1]

        return probs, action


    # TODO: update the temperature somewhere (first 30 moves of each game, T=1)
    def search(self, game, node, temp):
        self.root = node
        self.game = game

        # do NB_SIM of monte carlo
        for i in range(NB_SIM):
            node = self.root
            # do we need to create a clone at each iteration?

            # loop till node have childrens (till game not finished)
            while not node.is_leaf():
                Q_P_N = [np.array([[child_node.Q_sa, child_node.P_sa, child_node.N_sa] for child_node in node.childrens])]
                best_action = select_action(Q_P_N)
                node = node.childrens[best_action]

                # use that action in the game
                # TODO: implement step in game
                state, done = game.play_action(node.action)

            # TODO: add rotation somewhere

            # get probabilities P_s and values from the network for this state
            P_s, v = self.net.predict(state)

            # Add Dirichlet noise to the root node
            if not node.parent:
                P_s = dirichlet_noise(P_s)

            # get available actions
            legal_actions = game.get_legal_actions()

            # get illegal actions: all actions - available actions
            illegal_actions = np.setdiff1d(np.arange(game.board_size ** 2 + 1), np.array(legal_actions))

            # renormalize the P_s vector P_s = [P(s, a1), P(s, a2), ..., P(s, a2)]
            # with P(s, ai) scalar
            P_s[illegal_actions] = 0
            P_s /= np.sum(P_s)

            # add children of the new node we are in if possible
            node.expand_nodes(P_s)

            # TODO: add this function
            terminate, W_sa = game.check_game_over(game.current_player)

            # Backpropagate node statistics from leaf to root node
            # TODO: put a negative sign - ?
            while not node:
                node.update(v) # v is defined line 93
                node = node.parent

        # see Play (Fig. 2d) p.8 of the paper
        # AlphaGo Zero selects a move a to play in the root position
        # proportional to its exponentiated visit count
        probs, action = self.select_move(temp)

        # the child node corresponding to the played
        # action becomes the new root node
        for child in self.root.childrens:
            if child.action == action:
                self.root = child
                break

        # TODO: what to return?
        return probas, action


