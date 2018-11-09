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
def sample_symmetry(final_state):
	"""
		Sample a random dihedral reflections or rotations
		of the state of the game `final_state`
	
	Args:
		final_state (np.array): State of the game: [n_rows, n_cols, 17]
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
		state (np.array): State of the game: [n_rows, n_cols, 17]
			according to the description under `Neural network architecture`
			from the paper page 8
	Returns:
		(np.array): All the dihedral transformations of the input state.
			Array of size: [8, n_rows, n_cols, 17] because we have
			8 dihedral transformations
	"""
	transformed_states = []

	for transform_str in transformations:
		transformed_states.append(dihedral_group[transform_str](state))

	return np.array(transformed_states)


def batch_symmetries(batch_states):
	"""
		Apply a random symmetry to each individual states in the batch.
		Each state is of dimension [n_rows, n_cols, n_stacks]
	Args:
		batch_states (np.array): batch of states: [batch_size, n_rows, n_cols, n_stacks]
	Returns:
		applied_transformations (np.array): random transformations (string) selected without replacement 
		trasnformed_batch_states (np.array): batch of states of dimension [batch_size, n_rows, n_cols, n_stacks]
			where each state [n_rows, n_cols, n_stacks] underwent a random transformation
	"""
	applied_transformations = np.random.choice(transformations, batch_states.shape[0])
	trasnformed_batch_states = np.array([dihedral_group[transformation](state)
											  for transformation, state in zip(applied_transformations, batch_states)])

	return applied_transformations, trasnformed_batch_states

def transform_pi(pi, transformation):
	"""
		Transform pi according to the transformation
	Args:
		pi (np.array): array of size [batch_size, n_rows * n_cols + 1]
		transformations (str): name of the transformation to apply to the policy `pi`
	Returns:
		transformed_pi (np.array): name of the transformation to apply to the policy `pi`
	"""
	transformed_pi = np.copy(pi)
	transformed_pi[:-1] = dihedral_group[transformation](pi[:-1].reshape(config.n_rows, config.n_cols)).flatten()
	return transformed_pi

