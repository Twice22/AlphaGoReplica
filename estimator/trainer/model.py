import numpy as np
import os

import symmetries
import tensorflow as tf
import config

import functools


@functools.lru_cache(maxsize=1)
def get_vars():
    """
        Returns all the variables from the files `config.py` as a dictionary
    """
    d = {k: v for k, v in config.__dict__.items() if not (k.startswith('__') or k.startswith('_'))}
    return d


def get_inputs():
    """
        Create the placeholders for the features and the labels
    """
    features = tf.placeholder(
        dtype=tf.float32,
        shape=[None, config.n_rows, config.n_cols, config.history * 2 + 1],
        name='x')

    # +1 for the terminal state
    labels = {'pi': tf.placeholder(tf.float32, [None, config.n_rows * config.n_cols + 1]),
              'v': tf.placeholder(tf.float32, [None])}

    return features, labels


def create_estimator(work_dir=None):
    """
    Create a custom estimator based on model_fn

    Args:
        work_dir (str): Path where to save the checkpoints, the model, ...

    Returns:
        Estimator
    """
    run_config = tf.estimator.RunConfig(
        save_summary_steps=config.summary_steps,
        keep_checkpoint_max=config.keep_checkpoint_max
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=get_vars(),
                                       config=run_config,
                                       model_dir=config.job_dir if work_dir is None else work_dir)

    return estimator


# see equation (1) of the paper page 2
def compute_loss(pi, z, p, v):
    # pi and p have size [batch_size, #positions] and tf.looses.softmax_cross_entropy
    # returns a tensor of size [batch_size] then take the average to have the
    # cross entropy value over the whole batch
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=pi,
                                                                   logits=p))

    # return a scalar
    mean_square = config.mean_square_weight * tf.losses.mean_squared_error(labels=z,
                                                                           predictions=v)

    regularization = tf.losses.get_regularization_loss()

    return cross_entropy, mean_square, regularization


# features comes from input_fn (in train.py) that reads the tf_records
def model_fn(features, labels, mode, params):
    """
    Create the model structure and compute the output

    Args:
        features (dict): dictionary that maps the key `x` to the
            tensor representing the game: [config.n_rows, config.n_cols, config.history * 2 + 1]
        labels (dict): dictionary that maps the keys `pi`and `x` to their values
            `pi`: [batch_size, config.n_rows * config.n_cols + 1]
            `z`: [batch_size, 1]
        mode (tf.estimator.Modekeys): Tensorflow mode to use (TRAIN, EVALUATE, PREDICT)
            needed in part because batch Normalization is different during training and
            testing phase
        params (dict): extra parameters of the Neural Net coming from the `config.py` file
            created by the `task.py` file
    Returns:
        tf.estimator.EstimatorSpec with given properties:
        mode: same as the input mode
        predictions: dict of tensors
            {
                'p': [batch_size, config.n_rows * config.n_cols + 1],
                'v': [batch_size]
            }
        loss: the loss
        train_op: the training operation
        eval_metric_ops: dictionary of metrics to evaluate the network
            {
                'pol_cost': tf.metrics.mean(cross_entropy),
                'val_cost': tf.metrics.mean(mean_square),
                'reg_cost': tf.metrics.mean(regularization),
                'pol_entropy': tf.metrics.mean(pol_entropy),
                'loss': tf.metrics.mean(loss)
            }
    """

    state_size = params['history'] * 2 + 1  # 17 by default for the game of Go
    n_rows = params['n_rows']
    n_cols = params['n_cols']
    n_residual_blocks = params['n_res_blocks']
    c = params['l2_regularization']  # normalization cst: 1e-4 by default (p8 under Optimization)
    conv_width = params['conv_width']
    fc_width = params['fc_width']
    val_conv_width = params['val_conv_width']
    pol_conv_width = params['pol_conv_width']
    lr_boundaries = params['learning_rates_scheduler']
    lr_values = params['learning_rates']
    momentum = params['momentum_rate']
    summary_step = params['summary_step']  # Number of steps before we log summary scalars

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    evaluate = (mode == tf.estimator.ModeKeys.EVAL)
    predict = (mode == tf.estimator.ModeKeys.PREDICT)

    # features is our placeholder
    z = labels["v"]
    pi = labels["pi"]
    x = features

    # Add regularizer
    reg = tf.contrib.layers.l2_regularizer(c)

    # reshape X to 4-D tensor [batch_size, width, height, state_size]
    # x = tf.reshape(x, [-1, n_rows, n_cols, state_size])

    # See under `Neural network architecture` of the paper
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=conv_width,  # 256 by default in the paper
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        use_bias=False,
        kernel_regularizer=reg
    )

    # Batch Norm #1
    batch_norm1 = tf.layers.batch_normalization(
        epsilon=1e-5,
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
            filters=conv_width,  # 256 by default in the paper
            kernel_size=[3, 3],
            padding="same",
            strides=1,
            use_bias=False,
            kernel_regularizer=reg
        )

        # Batch normalization #1
        res_batch_norm1 = tf.layers.batch_normalization(
            epsilon=1e-5,
            inputs=res_conv1,
            training=training
        )

        # ReLu #1
        res_relu1 = tf.nn.relu(res_batch_norm1)

        # Convolution Layer #2
        res_conv2 = tf.layers.conv2d(
            inputs=res_relu1,
            filters=conv_width,  # 256 by default in the paper
            kernel_size=[3, 3],
            padding="same",
            strides=1,
            use_bias=False,
            kernel_regularizer=reg
        )

        # Batch normalization #2
        res_batch_norm2 = tf.layers.batch_normalization(
            epsilon=1e-5,
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
        filters=pol_conv_width,  # 2 by default in the paper,
        kernel_size=[1, 1],
        padding="same",
        strides=1,
        use_bias=False,
        kernel_regularizer=reg
    )

    pol_batch_norm = tf.layers.batch_normalization(
        epsilon=1e-5,
        inputs=pol_conv,
        training=training,
        center=False,
        scale=False
    )

    pol_relu = tf.nn.relu(pol_batch_norm)

    # 2 filters by default in pol_conv so (* 2) here
    pol_flatten_relu = tf.reshape(pol_relu, [-1, n_rows * n_cols * pol_conv_width])

    logits = tf.layers.dense(inputs=pol_flatten_relu,
                             units=n_rows * n_cols + 1,
                             kernel_regularizer=reg)

    p = tf.nn.softmax(logits, name='policy_head')

    ### Value Head ###
    val_conv = tf.layers.conv2d(
        inputs=res_input_layer,
        filters=val_conv_width,  # 1 by default in the paper
        kernel_size=[1, 1],
        padding="same",
        strides=1,
        use_bias=False,
        kernel_regularizer=reg
    )

    val_batch_norm = tf.layers.batch_normalization(
        epsilon=1e-5,
        inputs=val_conv,
        training=training,
        center=False,
        scale=False
    )

    val_relu1 = tf.nn.relu(val_batch_norm)

    # number of squares in the board
    val_flatten_relu = tf.reshape(val_relu1, [-1, val_conv_width * n_rows * n_cols])

    val_dense1 = tf.layers.dense(inputs=val_flatten_relu,
                                 units=fc_width,  # 256 by default in the paper
                                 kernel_regularizer=reg)

    val_relu2 = tf.nn.relu(val_dense1)

    # return a tensor of shape [batch_size, 1]
    val_dense2 = tf.layers.dense(inputs=val_relu2,
                                 units=1,
                                 kernel_regularizer=reg)

    # need to flatten it to have shape [batch_size]
    flatten_val_dense2 = tf.reshape(val_dense2, [-1])

    v = tf.nn.tanh(flatten_val_dense2, name='value_head')

    cross_entropy, mean_square, regularization = compute_loss(pi, z, p, v)
    loss = cross_entropy + mean_square + regularization  # line 232

    # depending if we have loaded previous weights or not
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)

    # see the note in: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Compute evaluation metrics
    # Compute the policy entropy for each batch and average it over the size of the batch
    pol_entropy = -tf.reduce_mean(tf.reduce_sum(p * tf.log(p), axis=1))

    # TODO: tf.reshape( , [1])  before tf.metrics.mean ?
    metrics = {
        'pol_cost': tf.metrics.mean(cross_entropy),
        'val_cost': tf.metrics.mean(mean_square),
        'reg_cost': tf.metrics.mean(regularization),
        'pol_entropy': tf.metrics.mean(pol_entropy),
        'loss': tf.metrics.mean(loss)
    }

    # can just add some summaries to the model:
    # https://stackoverflow.com/questions/43782767/how-can-i-use-tensorboard-with-tf-estimator-estimator
    
    # Toggle should_record_summaries
    tf.contrib.summary.record_summaries_every_n_global_steps(
        summary_step,
        global_step=global_step
    )
    for name, op in metrics.items():
        tf.contrib.summary.scalar(name, op[1], step=global_step)

    if evaluate:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    
    if predict:
        predictions = {
            'p': p,
            'v': v,
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions)  # TODO: define export_outputs?

    # stochastic gradient decent with momentum=0.9
    # and learning rate annealing. See `Optimization`
    # part of the paper page 8
    assert training

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step
        )

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)  # TODO: define hooks?


class NeuralNetwork:
    def __init__(self, weights_file=None):
        self.weights = weights_file
        self.features = None
        self.predictions = None

        # see https://www.tensorflow.org/guide/using_gpu for GPU usage
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.sess = tf.Session(graph=tf.Graph(), config=config)
        self.init_graph()

    def load_weights(self, weights_file):
        """
            Load the weights from the weights_file. Used to
            load the weights of the updated version of the player
        """
        tf.train.Saver().restore(self.sess, weights_file)

    def save_weights(self, work_dir=None, filename="model.ckpt-0"):
        with self.sess.graph.as_default():
            tf.train.Saver().save(self.sess, os.path.join(config.job_dir if work_dir is None else work_dir,
                                                      filename))

    def init_graph(self):
        with self.sess.graph.as_default():
            features, labels = get_inputs()
            estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, params=get_vars())

            self.features = features
            self.predictions = estimator_spec.predictions

            # load weights if exist
            if self.weights is not None:
                self.load_weights(self.weights)
            else:
                self.sess.run(tf.global_variables_initializer())

    # TODO: adapt to be able to use a batch of game (or batch of state)
    # inference
    def run(self, game):
        # retrieve the [n_rows, n_cols, 17] features (17 is by default)
        # need a batch of size 1: [1, n_rows, n_cols, 17]
        features = np.array([game.state])

        if config.use_random_symmetry:
            transformations, features = symmetries.batch_symmetries(features)

        outputs = self.sess.run(self.predictions, feed_dict={self.features: features})
        probs, values = outputs['p'], outputs['v']

        # need to retrieve the probabilities 
        # associate to the real state of the game
        if config.use_random_symmetry:
            probs = symmetries.unsymmetrize_pi(probs, transformations)

        return probs, values


def export_model(model_path, work_dir=None):
    """Take the latest checkpoint and copy it to model_path

    Args:
        model_path: The path to export the model
        work_dir: The path that contains the weights of the model
    """
    estimator = tf.estimator.Estimator(model_fn,
                                       model_dir=config.job_dir if work_dir is None else work_dir,
                                       params=get_vars())

    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
    for filename in all_checkpoint_files:
        ext = filename.partition(latest_checkpoint)[2]
        name = os.path.basename(latest_checkpoint)
        destination_path = os.path.join(model_path, name + ext)
        print("Copying {} to {}".format(filename, destination_path))
        tf.gfile.Copy(filename, destination_path)
