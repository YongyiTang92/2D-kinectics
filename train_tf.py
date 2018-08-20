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
        dataset = dataset.map(reader.parse_exmp)  # Sampling images from tfrecords
        if num_epochs is not None:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_siz, drop_remainder=True)
        dataset = dataset.shuffle(buffer_size=batch_size * 5)

        iterator = dataset.make_one_shot_iterator()
        batch_images, batch_labels = iterator.get_next()

        return batch_images, batch_labels


def parse_exmp(serial_exmp):
    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"id": tf.FixedLenFeature(
            [], tf.string),
            "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })
    feats = tf.parse_single_example(serial_exmp, features={'feature': tf.FixedLenFeature([], tf.string),
                                                           'label': tf.FixedLenFeature([10], tf.float32), 'shape': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(feats['feature'], tf.float32)
    label = feats['label']
    shape = tf.cast(feats['shape'], tf.int32)
    return image, label, shape


def build_graph(reader,
                model,
                train_data_pattern,
                num_classes=400,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):

    global_step = tf.Variable(0, trainable=False, name="global_step")
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size * num_towers,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = optimizer_class(learning_rate)

    batch_image, batch_labels = get_input_data_tensors(reader,
                                                       train_data_pattern,
                                                       batch_size=batch_size,
                                                       num_epochs=num_epochs,
                                                       num_readers=num_readers)
    tf.summary.image('input image', batch_image)

    tower_inputs = tf.split(batch_image, num_towers)
    tower_labels = tf.split(batch_labels, num_towers)
    tower_gradients = []
    tower_predictions = []
    tower_label_losses = []
    tower_reg_losses = []
    for i in range(num_towers):
        with tf.device(device_string % i):
            with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
                    result = model.create_model(
                        tower_inputs[i],
                        vocab_size=num_classes,
                        labels=tower_labels[i])
                    for variable in slim.get_model_variables():
                        tf.summary.histogram(variable.op.name, variable)

                    predictions = result["predictions"]
                    tower_predictions.append(predictions)

                    label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i])
                    reg_loss = tf.constant(0.0)
                    reg_losses = tf.losses.get_regularization_losses()
                    if reg_losses:
                        reg_loss += tf.add_n(reg_losses)

                    tower_reg_losses.append(reg_loss)

                    # Adds update_ops (e.g., moving average updates in batch normalization) as
                    # a dependency to the train_op.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    if update_ops:
                        with tf.control_dependencies(update_ops):
                            barrier = tf.no_op(name="gradient_barrier")
                            with tf.control_dependencies([barrier]):
                                label_loss = tf.identity(label_loss)
                    tower_label_losses.append(label_loss)
                    # Incorporate the L2 weight penalties etc.
                    final_loss = regularization_penalty * reg_loss + label_loss
                    gradients = optimizer.compute_gradients(final_loss,
                                                            colocate_gradients_with_ops=False)
                    tower_gradients.append(gradients)

    label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
    tf.summary.scalar("label_loss", label_loss)
    if regularization_penalty != 0:
        reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
        tf.summary.scalar("reg_loss", reg_loss)
    merged_gradients = utils.combine_gradients(tower_gradients)
    if clip_gradient_norm > 0:
        with tf.name_scope('clip_grads'):
            merged_gradients = utils.clip_gradient_norms(merged_gradients, clip_gradient_norm)

    train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
    tf.add_to_collection("input_batch_raw", model_input_raw)
    tf.add_to_collection("input_batch", batch_image)
    tf.add_to_collection("labels", tf.cast(batch_labels, tf.float32))
    tf.add_to_collection("train_op", train_op)


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
