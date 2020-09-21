import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD


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


def pretrain_generator(features_batch, real_labels, generator, verbose=False, weight_gaussian_loss=1,
                       optimizer=Adam(lr=0.001)):
    pass
