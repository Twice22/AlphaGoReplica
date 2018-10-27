import numpy as np
import functools

# see examples: https://en.wikipedia.org/wiki/Dihedral_group
dihedral_group = {
    'identity': lambda x: x,
    'rot90': lambda x: np.rot90(x, k=3), # 90 clockwise
    'rot180': lambda x: np.rot90(x, k=2),
    'rot270': lambda x: np.rot90(x),
    'reflect': lambda x: np.fliplr(x),
    'reflectrot90': lambda x: np.fliplr(np.rot90(x)), # 90 clockwise
    'reflectrot180': np.flipud,
    'reflectrot270': lambda x: np.fliplr(np.rot90(x, k=3))
}

transformations = list(dihedral_group.keys())

# See Expand and evaluate (Fig. 2b) page 8 of the paper
def sample_dihedral(final_state):
	"""
		Sample a random dihedral reflections or rotations
		of the state of the game `final_state`
	
	Args:
		final_state (np.array): State of the game: [board_size, board_size, 17]
			according to the description under `Neural network architecture` from
			the paper page 8
	Returns:
		(np.array): a random transformation of the state selected
			among all the dihedral reflections or rotations
	"""

	# select random transformation
	transform_str = np.random.choice(transformations)
	return dihedral_group[transform_str](final_state)

# See (4) Under `Domain knowledge` page 7 of the paper. This
# function is used to augment the dataset with all the dihedral
# transformations
def generate_random_symmetries(state):
	"""
		Create all the dihedral reflections/rotations of the
		state of the game
	Args:
		state (np.array): State of the game: [board_size, board_size, 17]
			according to the description under `Neural network architecture`
			from the paper page 8
	Returns:
		(np.array): All the dihedral transformations of the input state.
			Array of size: [8, board_size, board_size, 17] because we have
			8 dihedral transformations
	"""
	transformed_states = []

	for transform_str in transformations:
		transformed_states.append(dihedral_group[transform_str](state))

	return np.array(transformed_states)