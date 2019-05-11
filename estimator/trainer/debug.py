import tensorflow as tf
import numpy as np
from glob import glob
import os
import records
from config import *
import model

train_records = glob(os.path.join("data/selfplay_data", "*"))


# def _get_input():
#     return records.generate_input(
#         batch_size=FLAGS.train_batch_size,
#         records=train_records,
#         shuffle_records=False,
#         buffer_size=FLAGS.shuffle_buffer_size,
#         enable_transformations=False,
#         n_repeats=1
#     )
#
#
# train_input_fn = _get_input()
#
# with tf.Session() as sess:
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init_op)
#
#     while True:
#         try:
#             features, labels = sess.run(train_input_fn)
#             x = features
#             pi = labels['pi']
#             z = labels['v']
#
#             print(x.shape)
#             print(pi.shape)
#             print(pi[0])
#             print(np.sum(pi[0]))
#             print(z.shape)
#             # break
#         except tf.errors.OutOfRangeError:
#             break

model.export_model(FLAGS.job_dir, work_dir="temp")