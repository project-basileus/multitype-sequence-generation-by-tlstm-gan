import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_probability import distributions as tfd


def create_self_regression_data_batch(original_feature_sample, END_TOKEN=0):
    """
        Create self regression dataset given original sequences
        only support batch_size = 1 for now, aka, we have to pretrain the generator one seq by

        e.g. original event type seq: [1, 2, 3, 1, 1, 3]
        turns into a train dataset with the next token(softmax)/timstamp as the target

        [1, 0, 0, 0, 0, 0] -> 2 : [0, 0, 1.0, 0]
        [1, 2, 0, 0, 0, 0] -> 3 : [0, 0, 0, 1.0]
        [1, 2, 3, 0, 0, 0] -> 1 : [0, 1.0, 0, 0]
        [1, 2, 3, 1, 0, 0] -> 1 : [0, 1.0, 0, 0]
        [1, 2, 3, 1, 1, 0] -> 3 : [0, 0, 0, 1.0]

        same for timestamp sequence.

    :param original_feature_sample: tuple of numpy arrays (seq_et, seq_ts), each of shape (1, T, 1)
    :param END_TOKEN: by default 0. so that we can fill non-zero values in np.zeros
    :return:
    """
    orig_seq_et, orig_seq_ts = original_feature_sample
    _, T_orig, _ = orig_seq_et.shape

    self_regression_et = np.zeros((T_orig - 1, T_orig, 1))
    self_regression_ts = np.zeros((T_orig - 1, T_orig, 1))

    self_target_token = np.zeros((T_orig - 1, 1))
    self_target_timestamp = np.zeros((T_orig - 1, 1))

    for i in range(T_orig - 1):
        self_regression_et[i, :i + 1, :] = orig_seq_et[:, :i + 1, :]
        self_regression_ts[i, :i + 1, :] = orig_seq_ts[:, :i + 1, :]

        self_target_token[i, :] = orig_seq_et[:, i + 1, :]
        self_target_timestamp[i, :] = orig_seq_ts[:, i + 1, :]

    return self_regression_et, self_regression_ts, self_target_token, self_target_timestamp


def pretrain_discriminator(features_batch, real_labels, discriminator, verbose=False, weight_gaussian_loss=1,
                           optimizer=Adam(lr=0.001)):
    # train the discriminator
    with tf.GradientTape() as tape:
        real_et, real_ts = features_batch

        # train discriminator
        true_prob, gaussian_log, mask = discriminator((real_et, real_ts))

        # calculate masked neg-likelihood of gaussian mixture
        gaussian_log = gaussian_log[:, :, 0:1]
        gaussian_log = tf.boolean_mask(gaussian_log, mask)
        gaussian_loss = -tf.reduce_sum(gaussian_log)

        # cross-entropy loss
        ce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(real_labels, true_prob, from_logits=False))
        discriminator_loss = gaussian_loss * weight_gaussian_loss + ce_loss

        if verbose:
            print('discriminator token loss:{}'.format(ce_loss))
            print('discriminator gaussian loss:{}'.format(gaussian_loss))
            print('total discriminator loss:{}'.format(discriminator_loss))

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return ce_loss, gaussian_loss


def pretrain_generator(feature_sample, generator, verbose=False, weight_gaussian_loss=1,
                       optimizer=Adam(lr=0.001)):
    state_et_batch, state_ts_batch = feature_sample
    _, T, _ = state_et_batch.shape

    # train the generator
    with tf.GradientTape() as tape:
        generator.reset_states()
        ce_loss_list = []
        gaussian_list = []
        for i in range(0, T-1):
            curr_state_et = state_et_batch[:, i:i + 1, :]
            curr_state_ts = state_ts_batch[:, i:i + 1, :]
            target_et = state_et_batch[:, i + 1, :]
            target_ts = state_ts_batch[:, i + 1, :]

            token_prob, time_out = generator([curr_state_et, curr_state_ts])

            gaussian_log = time_out.log_prob(target_ts)
            gaussian_loss = -tf.reduce_mean(gaussian_log)  # one step across the whole batch
            gaussian_list.append(gaussian_loss)

            ce_losses = tf.keras.losses.sparse_categorical_crossentropy(target_et, token_prob)
            ce_loss = tf.reduce_mean(ce_losses)
            ce_loss_list.append(ce_loss)

        total_ce_loss = tf.reduce_sum(ce_loss_list)
        total_gaussian_loss = tf.reduce_sum(gaussian_list)
        total_loss = total_ce_loss + weight_gaussian_loss * total_gaussian_loss

    if verbose:
        print('ce_loss:{}, gaussian_loss:{}'.format(total_ce_loss, total_gaussian_loss))

    # apply gradient decent per batch
    grads = tape.gradient(total_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return total_ce_loss, total_gaussian_loss
