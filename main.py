from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_probability as tfp
import tensorflow as tf

tf.get_logger().setLevel('FATAL')

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(initial_distribution=initial_distribution,
                              transition_distribution=transition_distribution,
                              observation_distribution=observation_distribution,
                              num_steps=14)
mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
