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

        # stop generating once hitting end_token
        curr_state_ts = tf.where(cond_end_token, curr_state_ts, sampled_ts)
        all_state_ts = tf.concat([all_state_ts, curr_state_ts], axis=1)

    return all_state_et, all_state_ts, episode_token_probs, gaussian_log


def generate_sequences(N_gen, generator, batch_size, T, recover_to_timestamp=True):
    """
        Generate sequences batch per batch
    :param N_gen: total number of seqs to be generated
    :param generator:
    :param batch_size:
    :param T:
    :param recover_to_timestamp: whether to recover time deltas to absolute timestamps
    :return: a python list of shape [N_gen, T, 2]
    """
    all_type_seq = None
    all_time_seq = None
    N = 0

    while N < N_gen:
        batch_state_et, batch_state_ts, _, _ = generate_batch_sequence_by_rollout(generator, batch_size, T,
                                                                                  end_token=0, init_token=1.0,
                                                                                  max_time=1024, verbose=False)

        batch_type_seq = batch_state_et.numpy()
        batch_time_seq = batch_state_ts.numpy()

        # recover time delta to time stamps
        if recover_to_timestamp:
            batch_time_seq = np.cumsum(batch_time_seq, axis=1)

        if all_type_seq is None:
            all_type_seq = batch_type_seq
        else:
            all_type_seq = np.concatenate([all_type_seq, batch_type_seq], axis=0)

        if all_time_seq is None:
            all_time_seq = batch_time_seq
        else:
            all_time_seq = np.concatenate([all_time_seq, batch_time_seq], axis=0)

        N += batch_size

    # concat type and time in depth
    concated_seq_list = np.concatenate([all_type_seq, all_time_seq], axis=2).tolist()

    return concated_seq_list[:N_gen]


def train_generator(generator, discriminator, critic_network, batch_size, T, verbose=False,
                    weight_gaussian_loss=1,
                    optimizer=Adam(lr=0.001)):
    # reset hidden states for critic network per batch
    critic_network.reset_states()

    with tf.GradientTape(persistent=True) as tape:
        states_et, states_ts, episode_token_probs, gaussian_log = generate_batch_sequence_by_rollout(generator,
                                                                                                     batch_size, T,
                                                                                                     end_token=0,
                                                                                                     init_token=1.0,
                                                                                                     max_time=1024,
                                                                                                     verbose=False)
        ce_loss_list = []
        gaussian_list = []
        critic_loss_list = []

        # run disc on whole sequence
        # true_return is the total reward for generating this seq
        true_return = discriminator((states_et, states_ts))

        for i in range(0, T):
            # TODO: should we include the init token in loss?
            curr_state_et = states_et[:, i:i + 1, :]
            curr_state_ts = states_ts[:, i:i + 1, :]

            curr_token_prob = episode_token_probs[:, i:i + 1]
            curr_gaussian_log = gaussian_log[:, i:i + 1]

            q_value = critic_network([curr_state_et, curr_state_ts])
            diff = true_return - q_value

            # At this point in history, the critic estimated that we would get a
            # total reward = `q_value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `true_return`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.

            mask = tf.squeeze(curr_state_et)
            curr_state_et = tf.boolean_mask(curr_state_et, mask)
            curr_state_ts = tf.boolean_mask(curr_state_ts, mask)
            curr_token_prob = tf.boolean_mask(curr_token_prob, mask)
            curr_gaussian_log = tf.boolean_mask(curr_gaussian_log, mask)

            diff = tf.boolean_mask(diff, mask)

            ce_loss_list.append(-tf.reduce_mean(tf.math.log(curr_token_prob) * diff))
            gaussian_list.append(-tf.reduce_mean(curr_gaussian_log * diff))
            critic_loss_list.append(tf.keras.losses.MSE(true_return, q_value))

        total_ce_loss = tf.reduce_sum(ce_loss_list)
        total_gaussian_loss = tf.reduce_sum(gaussian_list)

        total_generator_loss = total_ce_loss + weight_gaussian_loss * total_gaussian_loss
        total_critic_loss = tf.reduce_sum(critic_loss_list)

        if verbose:
            print('generator token loss:{}'.format(total_ce_loss))
            print('generator gaussian loss:{}'.format(total_gaussian_loss))
            print('generator critic loss:{}'.format(total_critic_loss))

    # update generator
    generator_grads = tape.gradient(total_generator_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))

    # update critic network
    critic_grads = tape.gradient(total_critic_loss, critic_network.trainable_variables)
    optimizer.apply_gradients(zip(critic_grads, critic_network.trainable_variables))

    # explicitly drop tape because persistent=True
    del tape

    return total_ce_loss, total_gaussian_loss, total_critic_loss


def train_discriminator(features_batch, generator, discriminator, batch_size, T, verbose=False,
                        optimizer=Adam(lr=0.001)):
    # train the discriminator
    with tf.GradientTape() as tape:
        real_et, real_ts = features_batch
        real_labels = tf.ones((batch_size, 1))  # (batch_size, 1)

        generated_et, generated_ts, episode_token_probs, gaussian_log = generate_batch_sequence_by_rollout(generator,
                                                                                                           batch_size,
                                                                                                           T,
                                                                                                           end_token=0,
                                                                                                           init_token=1.0,
                                                                                                           max_time=1024,
                                                                                                           verbose=False)
        generated_labels = tf.zeros((batch_size, 1))

        total_et = tf.concat([generated_et, real_et], axis=0)
        total_ts = tf.concat([generated_ts, real_ts], axis=0)
        total_labels = tf.concat([generated_labels, real_labels], axis=0)

        # train discriminator
        true_prob = discriminator((total_et, total_ts))

        # cross-entropy loss
        ce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(total_labels, true_prob, from_logits=False))
        discriminator_loss = ce_loss

        if verbose:
            print('total discriminator loss:{}'.format(discriminator_loss))

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return ce_loss
