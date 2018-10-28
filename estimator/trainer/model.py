import os
import tensorflow as tf


def create_estimator(params, config, model_dir):
	"""
	Create a custom estimator based on _model_fn

	Args:
		params: parameters for the model
		config (tf.estimator.RunConfig): define the runtime environment for the estimator
		model_dir: from where to load the weights. TODO: do we keep that?
	Returns:
		Estimator
	"""

	# see equation (1) of the paper page 2
	def compute_loss(pi, z, p, v, c):
		cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=pi,
														logits=p)
		mean_square = tf.losses.mean_squared_error(labels=z,
												   predictions=v) # TODO: reshape?

		regularization = tf.losses.get_regularization_loss()

		return cross_entropy + mean_square + regularization


	def _model_fn(features, labels, mode, params):
		"""
		Args:
			features (dict): dictionary that maps the key `x` to the
				tensor representing the game: [batch_size, n_rows, n_cols, 17]
			labels (dict): dictionary that maps the keys `pi`and `x` to their values
				`pi`: [batch_size, n_rows * n_cols + 1]
				`v`: [batch_size]
			mode (tf.estimator.Modekeys): Tensorflow mode to use (TRAIN, EVALUATE, PREDICT)
				needed in part because batchnormalization is different during training and
				testing phase
			params (dict): extra parameters (TODO: add extra params?)
		Returns:
			tf.estimator.EstimatorSpec (TODO: add description)
		"""

		""" Create the model structure and compute the output """
		state_size = params.state_size # 17 for the game of Go
		action_size = params.action_size
		n_rows = params.n_rows
		n_cols = parasm.n_cols
		n_residual_blocks = params.n_residual_blocks
		c = params.c # normalization cst

		training = (mode == tf.estimator.ModeKeys.TRAIN)
		evaluate = (mode == tf.estimator.ModeKeys.TRAIN)
		predict = (mode == tf.estimator.ModeKeys.PREDICT)

		# TODO: features is our placeholder
		z = labels["z"]
		pi = labels["pi"]
		x = features["x"]

		# Add regularizer
		reg = tf.contrib.layers.l2_regularizer(c)

		# Input Layer
		# reshape X to 4-D tensor [batch_size, width, height, state_size]
		input_layer = tf.reshape(x, [-1, n_rows, n_cols, state_size])

		# See under `Neural network architecture` of the paper
		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filter=256,
			kernel_size=[3, 3],
			padding="same",
			strides=1,
			use_bias=False,
			kernel_regularizer=reg
		)

		# Batch Norm #1
		batch_norm1 = tf.layers.batch_normalization(
			inputs=conv1,
			training=training
		)

		# ReLU #1
		res_input_layer = tf.nn.relu(batch_norm1)

		# residual tower
		# `n_residual_blocks` residual blocks
		for i in range(n_residual_blocks):

			# Convolution Layer #1
			res_conv1 = tf.layers.conv2d(
				inputs=res_input_layer,
				filter=256,
				kernel_size=[3, 3],
				padding="same",
				strides=1,
				use_bias=False,
				kernel_regularizer=reg
			)

			# Batch normalization #1
			res_batch_norm1 = tf.layer.batch_normalization(
				inputs=res_conv1,
				training=training
			)

			# ReLu #1
			res_relu1 = tf.nn.relu(res_batch_norm1)

			# Convolution Layer #2
			res_conv2 = tf.layers.conv2d(
				inputs=res_relu1,
				filter=256,
				kernel_size=[3, 3],
				padding="same",
				strides=1,
				use_bias=False,
				kernel_regularizer=reg
			)

			# Batch normalization #2
			res_batch_norm2 = tf.layer.batch_normalization(
				inputs=res_conv2,
				training=training
			)

			# Skip connection
			res_batch_norm2 += res_input_layer

			# ReLu #2
			res_input_layer = tf.nn.relu(res_batch_norm2)


		### Policy Head ###
		pol_conv = tf.layers.conv2d(
			inputs=res_input_layer,
			filter=2,
			kernel_size=[1, 1],
			padding="same",
			strides=1,
			use_bias=False,
			kernel_regularizer=reg
		)

		pol_batch_norm = tf.layer.batch_normalization(
			inputs=pol_conv,
			training=training
		)

		pol_relu = tf.nn.relu(pol_batch_norm)

		# TODO: change filter for variables
		pol_flatten_relu = tf.reshape(relu, [-1, n_rows * n_cols * 2]) # 2 filters in pol_conv so * 2 here

		logits = tf.layers.dense(inputs=pol_flatten_relu,
								 units=action_size,
								 kernel_regularizer=reg)

		p = tf.nn.softmax(logits, name='policy_head')

		### Value Head ###
		val_conv = tf.layers.conv2d(
			inputs=res_input_layer,
			filter=1, # TODO: change for variables?
			kernel_size=[1, 1],
			padding="same",
			strides=1,
			use_bias=False,
			kernel_regularizer=reg
		)

		val_batch_norm = tf.layers.batch_normalization(
			inputs=val_conv,
			training=training
		)

		val_relu1 = tf.nn.relu(val_batch_norm)

		val_flatten_relu = tf.reshape(val_relu1, [-1, action_size * 1]) # if change for var, need to change 1 for var here

		val_dense1 = tf.layers.dense(inputs=val_flatten_relu,
									 units=256, # TODO: change for variables
									 kernel_regularizer=reg
		)

		val_relu2 = tf.nn.relu(val_dense1)

		# need a reshape?
		val_dense2 = tf.layers.dense(inputs=val_relu2,
									 units=1,
									 kernel_regularizer=reg
		)

		v = tf.nn.tanh(val_dense2, name='value_head')

		loss = compute_loss(pi, z, p, v, c)

		# stochastic gradient decent with momentum=0.9
		# and learning rate annealing. See `Optimization`
		# part of the paper page 8
		if training:
			optimizer = tf.train.MomentumOptimizer(
				learning_rate=learning_rate,
				momentum=momentum
			)

			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step()
			)

			return tf.estimator.EstimatorSpec(
				mode,
				loss=loss) # TODO: define hooks?

		if predict:
			predictions = {
				'p': p, # TODO: maybe need p[0]
				'v': v # TODO: maybe need v[0][0]
			}

			return tf.estimator.EstimatorSpec(
				mode,
				predictions=predictions) # TODO: define export_outputs?