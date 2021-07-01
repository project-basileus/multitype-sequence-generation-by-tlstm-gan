import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_probability import distributions as tfd


def pretrain_discriminator(event_type_batch, time_delta_batch, label_batch, discriminator, verbose=False, optimizer=Adam(lr=0.001)):
    # train the discriminator
    with tf.GradientTape() as tape:
        # train discriminator
        true_prob = discriminator([event_type_batch, time_delta_batch])

        # cross-entropy loss
        discriminator_loss = ce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(label_batch, true_prob, from_logits=False)
        )

        if verbose:
            print('total discriminator loss:{}'.format(discriminator_loss))

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return discriminator_loss


def pretrain_discriminator_and_critic(event_type_batch, time_delta_batch, label_batch, discriminator, critic, verbose=False, optimizer=Adam(lr=0.001)):
    # train the discriminator
    with tf.GradientTape(persistent=True) as tape:
        # train discriminator
        true_prob = discriminator([event_type_batch, time_delta_batch])

        # cross-entropy loss
        discriminator_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(label_batch, true_prob, from_logits=False)
        )
        
        # train critic step by step
        critic_step_loss = []
        critic.reset_states()
        for i in range(T):
            curr_state_et = event_type_batch[:, i:i+1, :]
            curr_state_ts = time_delta_batch[:, i:i+1, :]
            
            q_value = critic([curr_state_et, curr_state_ts])
            critic_mxe_loss = tf.keras.losses.MSE(true_prob, q_value)
            critic_step_loss.append(tf.reduce_mean(critic_mxe_loss))

        critic_loss = tf.reduce_mean(critic_step_loss)

        if verbose:
            print('total discriminator loss:{}'.format(discriminator_loss))
            print('total critic loss:{}'.format(critic_loss))

    disc_grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
    del tape
    
    return discriminator_loss, critic_loss


def pretrain_generator(event_type_batch, time_delta_batch, generator, verbose=False, optimizer=Adam(lr=0.001)):
    _, T, _ = event_type_batch.shape
    
    # train the generator
    with tf.GradientTape() as tape:
        generator.reset_states()
        step_token_ce_loss_list = []
        step_time_gaussian_loss_list = []

        for i in range(0, T - 1):
            curr_state_et = event_type_batch[:, i:i + 1, :]
            curr_state_ts = time_delta_batch[:, i:i + 1, :]
            
            target_et = event_type_batch[:, i + 1, :]
            target_ts = time_delta_batch[:, i + 1, :]            
            
            token_logits, time_mu, time_sigma = generator([curr_state_et, curr_state_ts])

            token_ce_losses = tf.keras.losses.categorical_crossentropy(target_et, token_logits, from_logits=True)
            token_ce_loss = tf.reduce_mean(token_ce_losses)
            step_token_ce_loss_list.append(token_ce_loss)
            
            time_dist = tfp.distributions.Normal(time_mu, time_sigma)            
            time_gaussian_log = time_dist.log_prob(target_ts)
            time_gaussian_loss = -tf.reduce_mean(time_gaussian_log)  # one step across the whole batch
            step_time_gaussian_loss_list.append(time_gaussian_loss)
    
        episode_token_ce_loss = tf.reduce_mean(step_token_ce_loss_list)
        episode_time_gaussian_loss = tf.reduce_mean(step_time_gaussian_loss_list)
        generator_loss = episode_token_ce_loss + episode_time_gaussian_loss

    if verbose:
        print('token ce loss:{}'.format(episode_token_ce_loss))
        print('time gaussian loss:{}'.format(episode_time_gaussian_loss))
        print('train loss:{}'.format(generator_loss))

    # apply gradient decent per batch
    grads = tape.gradient(generator_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return generator_loss, episode_token_ce_loss, episode_time_gaussian_loss