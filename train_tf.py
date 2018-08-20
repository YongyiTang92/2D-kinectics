import argparse
import os
import socket
import pdb
import time
import numpy as np
from utils import accuracy, AverageMeter, save_checkpoint, get_mean_size
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow import gfile
from tensorflow import flags
from tensorflow import logging

FLAGS = flags.FLAGS

if __name__ == "__main__":
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string("input", "rgb", "input image type")

    # settings
    flags.DEFINE_integer("num_epochs", 5,
                         "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer("max_steps", None,
                         "The maximum number of iterations of the training loop.")
    flags.DEFINE_integer("export_model_steps", 10000,
                         "The period, in number of steps, with which the model "
                         "is exported for batch prediction.")

    # hyper params
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_float(
        "regularization_penalty", 1.0,
        "How much weight to give to the regularization loss (the label loss has "
        "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.0001,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.9,
                       "Learning rate decay factor to be applied every "
                       "learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                       "Multiply current learning rate by learning_rate_decay "
                       "every learning_rate_decay_examples.")

    # Other flags.
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    flags.DEFINE_bool(
        "log_device_placement", False,
        "Whether to write the device on which every op will run into the "
        "logs on startup.")


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
    """Creates the section of the graph which reads the training data.

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      num_readers: How many I/O threads to use.

    Returns:
      A tuple containing the image tensor, labels tensor

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for training.")
    with tf.name_scope("train_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find training files. data_pattern='" +
                          data_pattern + "'.")
        logging.info("Number of training files: %s.", str(len(files)))
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_readers)
        dataset = dataset.map()  # Sampling images from tfrecords
        if num_epochs is not None:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=batch_size * 5)

        iterator = dataset.make_one_shot_iterator()
        batch_images, batch_labels = iterator.get_next()

        return batch_images, batch_labels


def build_graph():
    global_step = tf.Variable(0, trainable=False, name="global_step")


class Trainer(object):
    def __init__(self):
        pass

    def run(self):
        sv = tf.train.Supervisor(
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=0,
            save_summaries_secs=120,
            saver=saver)

        logging.info("%s: Starting managed session.", task_as_string(self.task))
        with sv.managed_session(target, config=self.config) as sess:
            try:
                logging.info("%s: Entering training loop.", task_as_string(self.task))
                while (not sv.should_stop()) and (not self.max_steps_reached):
                    sess.run(train_op)
            except tf.errors.OutOfRangeError:
                logging.info("%s: Done training -- epoch limit reached.",
                             task_as_string(self.task))

        logging.info("%s: Exited training loop.", task_as_string(self.task))
        sv.Stop()


if __name__ == "__main__":
    app.run()
