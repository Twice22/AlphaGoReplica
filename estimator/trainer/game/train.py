import logging

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


class UpdateRatioSessionHook(tf.train.SessionRunHook):
    """A hook that computes ||grad|| / ||weights|| with frobenius norm"""


def train(*tf_records):
    """
    Train on examples
    """

    def _get_input():
        return records.generate_input(
            batch_size=config.train_batch_size,
            records=tf_records,
            shuffle_records=True,
            buffer_size=1000,  # TODO: use a parameter
            enable_transformations=True,
            n_repeats=1
        )


    estimator = model.create_estimator()
    batch_size = config.train_batch_size
    steps = config.steps  # TODO: define number of steps for training

    logging.info("Training, steps = %s, batch = %s -> %s examples",
                 steps or '?', batch_size,
                 (steps * batch_size) if steps else '?')

    # input_fn: provides input data as mini batches
    # hooks: list of subclass of tf.train.SessionRunHook
    try:
        estimator.train(input_fn=_get_input(),
                        hooks=[],  # TODO: add hooks
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
