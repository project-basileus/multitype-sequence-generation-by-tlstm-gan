import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD


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


def pretrain_generator(feature_sample, generator, event_vocab_dim, verbose=False, weight_gaussian_loss=1,
                       optimizer=Adam(lr=0.001)):
    self_regression_et, self_regression_ts, self_target_token, _ = create_self_regression_data_batch(feature_sample)
    # self_target_timestamp is not actually needed here,
    # because we cauculate log-likelihood of gaussian mixture fitting original input timestamps
    # rather than comparing the next actual timestamp with a sampled timestamp from the updated gm distribution
    N_reg = self_regression_et.shape[0]

    ce_loss_list = []
    gaussian_loss_list = []

    # train the generator
    with tf.GradientTape() as tape:
        for i in range(N_reg):
            curr_state_et = self_regression_et[[i], :, :]
            curr_state_ts = self_regression_et[[i], :, :]

            curr_target_token = int(self_target_token[i].item())
            curr_target_token_prob = np.zeros((event_vocab_dim,))
            curr_target_token_prob[curr_target_token] = 1.0

            pred_token_prob, gaussian_log, mask, alpha, mu, sigma = generator([curr_state_et, curr_state_ts])

            gaussian_log = gaussian_log[0, 0:i + 1, 0]  # masked to the current step only
            gaussian_loss = -tf.reduce_sum(gaussian_log)
            ce_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(
                curr_target_token_prob, pred_token_prob, from_logits=False, label_smoothing=0))

            ce_loss_list.append(ce_loss)
            gaussian_loss_list.append(gaussian_loss)

        ce_loss_batch = tf.reduce_mean(ce_loss_list)
        gaussian_loss_batch = tf.reduce_mean(gaussian_loss_list)
        pretrain_generator_loss_batch = ce_loss_batch + weight_gaussian_loss * gaussian_loss_batch

        if verbose:
            print('pretrain generator categorical cross-entropy loss:{}'.format(ce_loss_batch))
            print('pretrain generator gaussian loss:{}'.format(gaussian_loss_batch))

    # apply gradient decent per batch
    grads = tape.gradient(pretrain_generator_loss_batch, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return ce_loss_batch, gaussian_loss_batch
