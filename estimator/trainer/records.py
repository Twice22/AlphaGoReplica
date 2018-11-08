import tensorflow as tf
import numpy as np

import multiprocessing
import config

# see tutorials
# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

# convert data to features
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(features, pi, v):
	"""
	Args:
		features (np.array): dimension [n_rows, n_cols, n_stacks] of uint8
			by default for the Go game n_stacks = 17
		pi (np.array): label 1 representing the policy
			of dimension [n_rows * n_cols + 1] of float32
		v (float): label 2 representing the value
	"""
	features = {
		'x': _bytes_features(features.toString()),
		'pi': _bytes_features(pi.toString()),
		'v': _float_features(v)
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))

	return example

def write_records(filename, examples):
	"""
	Args:
		filename (str): name of the file to write to
		examples (tf.train.Example): example protocol buffer
			to write in the file named `filename`
	"""
	writer = tf.python_io.TFRecordWriter(filename)
	for example in examples:
		writer.write(example.SerializeToString())

	writer.close()

def read_records(batch_size, records, shuffle_records=True, shuffle_examples=True, n_repeats=1):

	num_threads = multiprocessing.cpu_count() if multi_threading else 1

    records = list(records)
    if shuffle_records:
        random.shuffle(records)

    record_list = tf.data.Dataset.from_tensor_slices(records)

    dataset = record_list.interleave(lambda x:
                                     tf.data.TFRecordDataset(
                                         x, compression_type='GZIP'),
                                     	 cycle_length=64, block_length=16)

    # see example at the end of 
    # https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays
    def parser(record):
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
		# tensor of shape [board_size, board_size, 17]
		x = decode_raw(d['x'], tf.uint8)
		x = tf.reshape(x, [batch_size,
						   config.n_rows,
						   config.n_cols,
						   (config.history + 1) * 2 + 1])
		x = tf.cast(x, tf.float32) # need float32 to perform calculations in the Neural network

		# convert pi from a scalar string tensor to a float32
		# and reshape the tensor
		pi = decode_raw(d['pi'], tf.float32)
		pi = tf.reshape(pi, [batch_size, config.n_rows * config.n_cols + 1])

		# d['v'] is already a float32 we just need to reshape it
		# to have [batch_size] dimension
		v = tf.reshape(d['v'], [batch_size])

		return x, {'pi': pi, 'v': v}

	dataset = dataset.map(parser, num_parallel_calls=num_threads)
	if shuffle_examples:
		dataset = dataset.shuffle(buffer_size=1000) # define buffer_size
	dataset = dataset.repeat(n_repeats).batch(batch_size)

	return dataset

def generate_input(batch_size, records, shuffle_records=True, shuffle_examples=True, n_repeats=1):
	dataset = read_records(
		batch_size,
		records,
		shuffle_records,
		shuffle_examples,
		n_repeats
	)

	# TODO: add dihedral transformations?

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()