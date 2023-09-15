"""Main script for running the LALME model."""

import os
import warnings

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags
import tensorflow as tf

import train_flow
import train_vmp_flow


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):

  # log to a file
  if FLAGS.log_dir:
    if not os.path.exists(FLAGS.log_dir):
      os.makedirs(FLAGS.log_dir)
    logging.get_absl_handler().use_absl_log_file()

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  logging.info('JAX device count: %r', jax.device_count())

  if FLAGS.config.method == 'flow':
    train_flow.train_and_evaluate(config=FLAGS.config, workdir=FLAGS.workdir)
  elif FLAGS.config.method == 'vmp_flow':
    train_vmp_flow.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.method == 'mcmc':
    sample_mcmc.sample_and_evaluate(config=FLAGS.config, workdir=FLAGS.workdir)
  else:
    raise ValueError(f'Unknown method {FLAGS.config.method}')


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
