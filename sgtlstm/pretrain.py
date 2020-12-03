import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_probability import distributions as tfd


def pretrain_discriminator(features_batch, real_labels, discriminator, verbose=False, optimizer=Adam(lr=0.001)):
    # train the discriminator
    with tf.GradientTape() as tape:
        real_et, real_ts = features_batch

        # train discriminator
        true_prob, _ = discriminator((real_et, real_ts))

        # cross-entropy loss
        ce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(real_labels, true_prob, from_logits=False))
        discriminator_loss = ce_loss

        if verbose:
            print('total discriminator loss:{}'.format(discriminator_loss))

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return ce_loss


def pretrain_generator(feature_sample, generator, verbose=False, weight_gaussian_loss=1,
                       optimizer=Adam(lr=0.001)):
    state_et_batch, state_ts_batch = feature_sample
    _, T, _ = state_et_batch.shape

    # train the generator
    with tf.GradientTape() as tape:
        generator.reset_states()
        ce_loss_list = []
        gaussian_list = []
        for i in range(0, T - 1):
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

        total_ce_loss = tf.reduce_mean(ce_loss_list)
        total_gaussian_loss = tf.reduce_mean(gaussian_list)
        total_loss = total_ce_loss + weight_gaussian_loss * total_gaussian_loss

    if verbose:
        print('train ce_loss:{}, train gaussian_loss:{}'.format(total_ce_loss, total_gaussian_loss))

    # apply gradient decent per batch
    grads = tape.gradient(total_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return total_ce_loss, total_gaussian_loss
