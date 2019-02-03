"""Train a network.
Usage:
  python train.py tfrecord1 tfrecord2 tfrecord3
"""


import logging

import numpy as np
import tensorflow as tf
import trainer.game.model as model
import trainer.game.config as config
import trainer.game.records as records


# Note: StepCounterHook inherits from SessionRunHook
# So we can define a subclass of StepCounterHook and pass
# it to the train function of the estimator model
class DisplayStepsPerSecond(tf.train.StepCounterHook):
    """hook that LOGS steps per second"""

    # Note: we redefine the _log_and_record method
    # See: https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/training/basic_session_run_hooks.py#L674
    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        s_per_sec = elapsed_steps / elapsed_time
        logging.log("{}: {:.3f} steps per second".format(global_step, s_per_sec))
        super()._log_and_record(elapsed_steps, elapsed_time, global_step)


def compute_update_ratio(weight_tensors, before_weights, after_weights):
    """computes ||grad|| / ||weights|| with Frobenius norm"""
    # before and after_weights are a list of weights
    deltas = [after - before for after, before in zip(after_weights, before_weights)]
    delta_norms = [np.linalg.norm(d.ravel()) for d in deltas]
    weight_norms = [np.linalg.norm(w.ravel()) for w in before_weights]
    ratios = [d / w for d, w in zip(delta_norms, weight_norms)]
    all_summaries = [
        tf.Summary.Value(tag='update_ratios/' +
                             tensor.name, simple_value=ratio)
        for tensor, ratio in zip(weight_tensors, ratios)]
    return tf.Summary(value=all_summaries)


class UpdateRatioSessionHook(tf.train.SessionRunHook):
    """A hook that computes ||grad|| / ||weights|| with Frobenius norm"""

    def __init__(self, output_dir, every_n_steps=1000):
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self.before_weights = None
        self.file_writer = None
        self.weight_tensors = None
        self.global_step = None

    def begin(self):
        # begin is called before using the session and can add ops to the graph
        # then the graph is frozen and cannot be modify

        # return the FileWriter for the specified directory
        self.file_writer = tf.summary.FileWriterCache.get(self.output_dir)

        # return a list of Variable objects (Weights of the network)
        self.weight_tensors = tf.trainable_variables()

        # get (or create if not exist) the global_step tensor
        self.global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        # run_context is a SessionRunContext. It contains info
        # about the TF session and the  op/tensors requested

        # called before `run()` to add ops/tensors to the `run()` call
        # or call run inside attach to the run_context to get the values
        # passed to the session
        global_step = run_context.session.run(self.global_step)
        if global_step % self.every_n_steps == 0:
            self.before_weights = run_context.session.run(self.weight_tensors)

    def after_run(self, run_context, run_values):
        # run_values: contains results of requested ops/tensors by `before_run()`
        # here we don't use run_values becausse `before_run()` returns None
        # We just fetch the values again after `run()` (so the values have been updated)
        global_step = run_context.session.run(self.global_step)
        if self.before_weights is not None:
            after_weights = run_context.session.run(self.weight_tensors)
            weight_update_summaries = compute_update_ratio(
                self.weight_tensors, self.before_weights, after_weights
            )

            # add_summary takes the result of evaluating any summary op
            # (using tf.Session.run or tf.Tensor.eval). Can also pass tf.Summary
            self.file_writer.add_summary(
                weight_update_summaries, global_step
            )
            self.before_weights = None


def train(*tf_records):
    """
    Train on examples
    """

    def _get_input():
        return records.generate_input(
            batch_size=config.train_batch_size,
            records=tf_records,
            shuffle_records=True,
            buffer_size=config.shuffle_buffer_size,
            enable_transformations=True,
            n_repeats=1
        )

    hooks = [DisplayStepsPerSecond(config.work_dir),
             UpdateRatioSessionHook(output_dir=config.work_dir)]

    estimator = model.create_estimator()
    batch_size = config.train_batch_size
    steps = config.steps_to_train

    logging.info("Training, steps = %s, batch = %s -> %s examples",
                 steps or '?', batch_size,
                 (steps * batch_size) if steps else '?')

    # input_fn: provides input data as mini batches
    # hooks: list of subclass of tf.train.SessionRunHook
    try:
        estimator.train(input_fn=_get_input(),
                        hooks=hooks,
                        steps=config.steps)  # TODO: nb of steps to train the model
    except ValueError as e:
        raise e


def main(argv):
    """
    Train on examples and export the model weights
    """
    tf_records = argv[1:]
    logging.info("Training on %s records: %s to %s",
                 len(tf_records), tf_records[0], tf_records[-1])

    # *tf_records is a list
    train(*tf_records)

    # TODO: define the export_path variable
    if config.export_path:
        model.export_model(config.export_path)


if __name__ == '__main__':
    main()
