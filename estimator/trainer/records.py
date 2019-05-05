import tensorflow as tf

import multiprocessing
import symmetries
import random
import config


# see tutorials
# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

# convert data to features
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(x, pi, v):
    """
    Args:
        x (np.array): dimension [n_rows, n_cols, n_stacks] of uint8
            by default for the Go game n_stacks = 17
        pi (np.array): label 1 representing the policy
            of dimension [n_rows * n_cols + 1] of float32
        v (float): label 2 representing the value (winner)
    """
    features = {
        'x': _bytes_feature(x.tostring()),
        'pi': _bytes_feature(pi.tostring()),
        'v': _float_feature(v)
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


def write_records(filename, examples):
    """
    Args:
        filename (str): name of the file to write to
        examples (tf.train.Example): example protocol buffer
            to write in the file named `filename`
    """
    writer = tf.python_io.TFRecordWriter(
        filename,
        options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.ZLIB))
    for example in examples:
        writer.write(example.SerializeToString())

    writer.close()


def read_records(batch_size, records, shuffle_records=True, buffer_size=1000, n_repeats=1,
                 multi_threading=True):
    """
    Args:
        batch_size (int): batch_size of data to return
        records (tuple[str]): tuple of tf.records filename
        shuffle_records (bool): Whether to shuffle the order of the files to read
        buffer_size (int): size of the buffer before we shuffle it
        n_repeats (int): how many times the data should be repeated. Default to 1
        multi_threading (bool): Whether to use multi threading or not
    Returns:
        (tf.Dataset): a dataset of batched tensors
    """
    n_threads = multiprocessing.cpu_count() if multi_threading else 1

    # transform tuple of records to list of records
    records = list(records)

    # shuffle the filename to read
    if shuffle_records:
        random.shuffle(records)

    # create a dataset from the input records.
    # if records = ["filename_1", "filename_2", "filename_3"]
    # then:
    # dataset = tf.data.Dataset.from_tensor_slices(x)
    # iter = dataset.make_one_shot_iterator()
    # el = iter.get_next()
    #
    # with tf.Session() as sess:
    #     print(sess.run(el))
    # returns b'filename_1'
    record_list = tf.data.Dataset.from_tensor_slices(records)

    # cycle_length: number of elements from the dataset that will be processed concurrently
    # idea: shuffle both the order of files being read (random.shuffle(records))
    # and the example being read from each file => (need sloppy=True)
    # https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave
    dataset = record_list.apply(tf.data.experimental.parallel_interleave(lambda x:
                                     tf.data.TFRecordDataset(
                                         x, compression_type='ZLIB'),  # same compression used during writing
                                     cycle_length=64, sloppy=True))

    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # ensure to have dataset whose batch_size = train_batch_size (drop_remainder=True)
    # if we play N games and if each game have more than batch_size actions
    # then only the end of the last game won't be use to train the network
    dataset = dataset.repeat(n_repeats).batch(batch_size, drop_remainder=True)

    # see example at the end of
    # https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays
    def parser(batch_of_examples):
        # FixedLenFeature is used to parse a fixed-length
        # input feature. Here we know for sure that all features
        # are always pass together so we don't need to pass
        # a `default_value`. See examples here:
        # https://www.tensorflow.org/api_docs/python/tf/parse_example
        features = {
            'x': tf.FixedLenFeature([], tf.string),
            'pi': tf.FixedLenFeature([], tf.string),
            'v': tf.FixedLenFeature([], tf.float32)
        }

        # return a dict that maps feature keys to Tensor
        d = tf.parse_example(batch_of_examples, features)

        # convert from a scalar string tensor to an uint8
        # tensor of shape [batch_size, config.n_rows, config.n_cols, 17]
        x = tf.decode_raw(d['x'], tf.uint8)
        x = tf.reshape(x, [batch_size,
                           config.n_rows,
                           config.n_cols,
                           config.history * 2 + 1])
        x = tf.cast(x, tf.float32)  # need float32 to perform calculations in the Neural network

        # convert pi from a scalar string tensor to a float32
        # and reshape the tensor
        pi = tf.decode_raw(d['pi'], tf.float32)
        pi = tf.reshape(pi, [batch_size, config.n_rows * config.n_cols + 1])

        # d['v'] is already a float32 we just need to reshape it
        # to have [batch_size] dimension
        v = d['v']
        v.set_shape([batch_size])

        return x, {'pi': pi, 'v': v}

    # as we batched the dataset before calling map, the map_func=parser
    # will be applied for each element of this dataset of size [batch_size]
    # Moreover each input will be a tf.Example (due to the use of
    # tf.data.experimental.parallel_interleave)
    dataset = dataset.map(
                parser,
                num_parallel_calls=n_threads)

    return dataset


# the arguments of this function are the returns values of the parser function
def _apply_transformations(input_x, output_dict):
    """
    Args:
        input_x (np.array): np.array of dimension [batch_size, n_rows, n_cols, n_stacks],
            where n_stacks = 17 for the Go game by default
        output_dict (dict): {'pi': pi, 'v': v} see the returned value of the parser function
    Returns:
        transformed_input_x (np.array): input_x that underwent random dihedral transformations
        output_dict (dict): same as output_dict beside output_dict["pi"] that is changed according to
            the transformations underwent by the input state `input_x`
    """
    def transform_py_function(x, pi):
        transformations, transformed_input_x = symmetries.batch_symmetries(x)
        transformed_pi = [symmetries.transform_pi(p, transformation) for p, transformation in
         zip(pi, transformations)]
        return transformed_input_x, transformed_pi

    # need to transform the policy `pi` to fit the transformed_input_x
    pi_tensor = output_dict['pi']
    transformed_x_tensor, transformed_pi_tensor = tuple(tf.py_func(  # TODO: use tf1.13 and tf.py_function
        transform_py_function,
        [input_x, pi_tensor],
        [input_x.dtype, pi_tensor.dtype]
    ))

    # need to set the shape of the tensor
    transformed_x_tensor.set_shape(input_x.get_shape())
    transformed_pi_tensor.set_shape(pi_tensor.get_shape())

    output_dict["pi"] = transformed_pi_tensor

    return transformed_x_tensor, output_dict


def generate_input(batch_size, records, shuffle_records=True, buffer_size=1000,
                   enable_transformations=False, n_repeats=1):
    """ Read tf.records and pass them to the neural network model"""
    dataset = read_records(
        batch_size,
        records,
        shuffle_records,
        buffer_size,
        n_repeats
    )

    if enable_transformations:
        dataset = dataset.map(_apply_transformations)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def make_selfplay_dataset(data_extracts):
    """
    Args:
        data_extracts (iter(tuples)): iterable of (pos, pi, result)
    Returns:
        tf_examples (iter(tf.Examples))
    """
    tf_examples = (create_example(x, pi, result) for x, pi, result in data_extracts)
    return tf_examples


# TODO: to define if we use Google Cloud
# see step 3: https://cloud.google.com/blog/big-data/2018/02/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine
def serving_input_receiver_fn():
    feature_tensor = {"x": tf.placeholder(dtype=tf.float32,
                                          shape=[None, config.n_rows, config.n_cols, config.history * 2 + 1],
                                          name='x')}
    return tf.estimator.export.ServingInputReceiver(feature_tensor, feature_tensor)
