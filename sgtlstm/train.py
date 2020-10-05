import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, LSTM, Embedding, Reshape, Dense
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from sgtlstm.TimeLSTM import TimeLSTM0, TimeLSTM1, TimeLSTM2, TimeLSTM3

tf.keras.backend.set_floatx('float64')


def generate_batch_sequence_by_rollout(
        G, batch_size, T, end_token=0, init_token=1.0, max_time=1024, verbose=False):
    # Begin from dummy init state (init_token=1, init_timestamp=0.0)
    curr_state_et = np.zeros([batch_size, 1, 1])
    curr_state_et[:, 0, 0] = init_token

    curr_state_ts = np.zeros([batch_size, 1, 1])
    curr_state_ts[:, 0, 0] = 0.0

    all_state_et = curr_state_et
    all_state_ts = curr_state_ts

    episode_token_probs = tf.constant(1., dtype=tf.float64, shape=(batch_size, 1))
    gaussian_log = tf.constant(0., dtype=tf.float64, shape=(batch_size, 1))

    G.reset_states()

    for step in range(1, T):  # sequence length
        token_prob, time_out = G([curr_state_et, curr_state_ts])

        sampled_et = tf.random.categorical(tf.math.log(token_prob), num_samples=1, dtype=tf.int32)
        sampled_et = tf.reshape(sampled_et, [batch_size, 1, 1]).numpy().astype(float)

        # get the chosen token probability per batch for each step
        sampled_et_indices = sampled_et.squeeze().astype(int).tolist()
        sampled_token_prob = token_prob.numpy()[np.arange(len(token_prob)), sampled_et_indices].reshape((batch_size, 1))
        episode_token_probs = tf.concat([episode_token_probs, sampled_token_prob], axis=1)

        # stop genererating once hit end_token
        cond_end_token = tf.equal(curr_state_et, end_token)
        curr_state_et = tf.where(cond_end_token, curr_state_et, sampled_et)
        all_state_et = tf.concat([all_state_et, curr_state_et], axis=1)

        # generate one timstamp using time_out
        sampled_ts_raw = time_out.sample()
        sampled_ts = tf.clip_by_value(tf.reshape(sampled_ts_raw, (batch_size, 1, 1))
                                      , clip_value_min=1, clip_value_max=max_time)

        # get the gaussian log likelihood for the sampled timestamps
        sampled_gaussian_log = time_out.log_prob(sampled_ts_raw)
        gaussian_log = tf.concat([gaussian_log, sampled_gaussian_log], axis=1)

        # stop genererating once hit end_token
        curr_state_ts = tf.where(cond_end_token, curr_state_ts, sampled_ts)
        all_state_ts = tf.concat([all_state_ts, curr_state_ts], axis=1)

    return all_state_et, all_state_ts, episode_token_probs, gaussian_log


# def generate_one_sequence_by_rollout(generator, T, event_vocab_dim, end_token=0, init_token=1, max_time=1024,
#                                      verbose=False):
#     # Begin from dummy init state (init_token=1, init_timestamp=0.0)
#     curr_state_et = np.zeros([T])
#     curr_state_et[0] = init_token
#     curr_state_et = curr_state_et.reshape((1, T, 1))
#
#     curr_state_ts = np.zeros([T])
#     curr_state_ts[0] = 0.0
#     curr_state_ts = curr_state_ts.reshape((1, T, 1))
#
#     # whole trajectory
#     states_et = (curr_state_et)
#     states_ts = (curr_state_ts)
#     episode_token_probs = tf.constant([1., ], dtype=tf.float64)
#
#     for step in range(1, T):  # sequence length
#         token_prob, gaussian_log, mask, alpha, mu, sigma = generator([curr_state_et, curr_state_ts])
#
#         # generate one timstamp using [alpha, mu, sigma]
#         gm = tfd.MixtureSameFamily(
#             mixture_distribution=tfd.Categorical(
#                 probs=alpha),
#             components_distribution=tfd.Normal(
#                 loc=mu,
#                 scale=sigma))
#
#         # sample next event token and time stamp
#         assert token_prob.shape == [1, event_vocab_dim]
#
#         sampled_et = tf.random.categorical(tf.math.log(token_prob), num_samples=1)
#         sampled_ts = tf.clip_by_value(gm.sample(), clip_value_min=1, clip_value_max=max_time)  # shape=[BATCH_SIZE,]
#
#         taken_action_idx = sampled_et.numpy().item()
#
#         if taken_action_idx == end_token:
#             if verbose:
#                 print('Generation ended early!')
#             break  # episode is over
#
#         taken_action_prob = token_prob[0][taken_action_idx]
#         taken_action_prob = tf.reshape(taken_action_prob, [1, ])
#         episode_token_probs = tf.concat([episode_token_probs, taken_action_prob], axis=0)
#
#         new_state_et = np.copy(curr_state_et)
#         new_state_ts = np.copy(curr_state_ts)
#
#         # TODO: 0 means 1 generation per batch
#         new_state_et[0, step, :] = sampled_et
#         new_state_ts[0, step, :] = sampled_ts
#
#         if verbose:
#             print('new_state_et', tf.squeeze(new_state_et))
#
#         states_et = np.concatenate((states_et, new_state_et))
#         states_ts = np.concatenate((states_ts, new_state_ts))
#
#         curr_state_et = new_state_et
#         curr_state_ts = new_state_ts
#         if verbose:
#             print('Generation done!')
#
#     if verbose:
#         print('episode length={}'.format(step + 1))
#         print('state_et =', states_et)
#         print('state_ts =', states_ts)
#         print('episode_token_probs =', episode_token_probs)
#         print('gaussian_log =', gaussian_log)
#
#     return states_et, states_ts, episode_token_probs, gaussian_log


def train_generator(generator, discriminator, T, event_vocab_dim, verbose=False, weight_gaussian_loss=1,
                    optimizer=Adam(lr=0.001)):
    with tf.GradientTape() as tape:
        states_et, states_ts, episode_token_probs, gaussian_log = generate_batch_sequence_by_rollout(generator,
                                                                                                     T, event_vocab_dim,
                                                                                                     verbose=verbose)

        # TODO: Does actual length need to be specified for each sample in batch?
        actual_length = episode_token_probs.shape[0]

        gaussian_log = gaussian_log[0, 0:actual_length, 0]
        true_prob, _, _ = discriminator((states_et, states_ts))
        token_loss = -tf.reduce_sum(tf.math.log(episode_token_probs) * true_prob)
        gaussian_loss = -tf.reduce_sum(gaussian_log * true_prob)
        generator_loss = token_loss + weight_gaussian_loss * gaussian_loss

        if verbose:
            print('generator token loss:{}'.format(token_loss))
            print('generator gaussian loss:{}'.format(gaussian_loss))
            print('total generator loss:{}'.format(generator_loss / actual_length))

    grads = tape.gradient(generator_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return token_loss, gaussian_loss


def train_discriminator(features_batch, generator, discriminator, T, event_vocab_dim, verbose=False,
                        weight_gaussian_loss=1,
                        optimizer=Adam(lr=0.001)):
    # feature_batch = (event_type, timestamp)
    batch_size = features_batch[0].shape[0]

    # train the discriminator
    with tf.GradientTape() as tape:
        real_et, real_ts = features_batch
        real_labels = tf.ones((real_et.shape[0], 1))  # (batch_size, 1)

        generated_et = tf.zeros([1, T, 1], dtype=tf.float64)
        generated_ts = tf.zeros([1, T, 1], dtype=tf.float64)
        for i in range(batch_size):
            states_et, states_ts, episode_token_probs, gaussian_log = generate_one_sequence_by_rollout(generator,
                                                                                                       T,
                                                                                                       event_vocab_dim,
                                                                                                       verbose=verbose)
            generated_et = tf.concat([generated_et, states_et[-1:, :, :]], axis=0)
            generated_ts = tf.concat([generated_ts, states_ts[-1:, :, :]], axis=0)
        generated_et = generated_et[1:, :, :]
        generated_ts = generated_ts[1:, :, :]
        generated_labels = tf.zeros((generated_et.shape[0], 1))

        total_et = tf.concat([generated_et, real_et], axis=0)
        total_ts = tf.concat([generated_ts, real_ts], axis=0)
        total_labels = tf.concat([generated_labels, real_labels], axis=0)

        # train discriminator
        true_prob, gaussian_log, mask = discriminator((total_et, total_ts))

        # calculate masked neg-likelihood of gaussian mixture
        gaussian_log = gaussian_log[:, :, 0:1]
        gaussian_log = tf.boolean_mask(gaussian_log, mask)
        gaussian_loss = -tf.reduce_sum(gaussian_log)

        # cross-entropy loss
        ce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(total_labels, true_prob, from_logits=False))
        discriminator_loss = gaussian_loss * weight_gaussian_loss + ce_loss

        if verbose:
            print('discriminator token loss:{}'.format(ce_loss))
            print('discriminator gaussian loss:{}'.format(gaussian_loss))
            print('total discriminator loss:{}'.format(discriminator_loss))

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return ce_loss, gaussian_loss
