import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

tf.keras.backend.set_floatx('float64')


def sample_gumbel(shape, eps=1e-20): 
    U = tf.random.uniform(shape,minval=0,maxval=1)   #gumbel noise
#     print('noise:{}'.format(U))
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=0.5): 
    y = logits + tf.cast(sample_gumbel(tf.shape(logits)), logits.dtype)
    return tf.nn.softmax( y / temperature)  # use softmax to approximate argmax

def gumbel_softmax(logits, temperature=0.5, hard=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    logits = tf.cast(logits, tf.float64)
    y = gumbel_softmax_sample(logits, temperature) # this is differentiable
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keepdims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def rollout_from_initial_gumbel(G, batch_size, init_et, init_ts, L=T, temperature=0.5, use_tlstm=False):
    # Begin from dummy init state (init_token=1, init_timestamp=0.0)
    all_state_et = init_et
    all_state_ts = init_ts
    
    l_prefix = init_et.shape[1]
    
    G.reset_states()
    
    if use_tlstm:
        # Time-LSTM
        G.layers[4].states[0] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        G.layers[4].states[1] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
    else:
        # LSTM-token
        G.layers[3].states[0] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        G.layers[3].states[1] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        # LSTM-time
        G.layers[4].states[0] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        G.layers[4].states[1] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))

    for _ in range(L + 1 - l_prefix):  # sequence length
        curr_state_et = all_state_et[:, -1:, :]
        curr_state_ts = all_state_ts[:, -1:, :]
        
        # add step noise to token and time inputs
#         curr_state_et = tf.cast(tf.where(curr_state_et == 1, 0.9, 0.1/3), tf.float64)
#         curr_state_ts = curr_state_ts + tf.random.normal(shape=tf.shape(curr_state_ts), mean=0.0, stddev=1, dtype=tf.float64)

        token_logits, time_delta_out = G([curr_state_et, curr_state_ts])
        
        # sample event types using Gumbel-softmax
        sampled_et = gumbel_softmax(token_logits, temperature)  # this is differentiable
        sampled_et = tf.reshape(sampled_et, [batch_size, 1, EVENT_VOCAB_DIM])        
        sampled_et = tf.cast(sampled_et, dtype=tf.float64) # cast sampled_et into float
        
        time_delta_out = tf.reshape(time_delta_out, [batch_size, 1, 1]) 
        
        # Do NOT stop genererating once hit end_token; G is supposed to learn it.
        all_state_et = tf.concat([all_state_et, sampled_et], axis=1)
        all_state_ts = tf.concat([all_state_ts, time_delta_out], axis=1)

    # the initlal random states are excluded in output
    return all_state_et[:, 1:, :], all_state_ts[:, 1:, :]


def generate_sequences_gumbel(N_gen, generator, batch_size, T, temperature=0.5, use_tlstm=False):
    """
        Generate sequences batch per batch
    :param N_gen: total number of seqs to be generated
    :param generator:
    :param batch_size:
    :param T:
    :return: a python list of shape [N_gen, T, 1]
    """
    N = 0
    all_type_seq = None
    all_time_seq = None
    
    init_token_noise, init_time_noise = generate_initial_noise(zero_time_start=False)
        
    while N < N_gen:
        batch_state_et, batch_state_ts = rollout_from_initial_gumbel(generator, batch_size, init_token_noise, init_time_noise, T, temperature, use_tlstm)

        if all_type_seq is None or all_time_seq is None:
            all_type_seq = batch_state_et
            all_time_seq = batch_state_ts
        else:
            all_type_seq = tf.concat([all_type_seq, batch_state_et], axis=0)
            all_time_seq = tf.concat([all_time_seq, batch_state_ts], axis=0)

        N += batch_size

    all_type_seq = all_type_seq[:N_gen, :, :]
    all_time_seq = all_time_seq[:N_gen, :, :]

    return all_type_seq, all_time_seq


def get_generation_metrics(G):
    _gen_seqs_et, _gen_seqs_ts = generate_sequences_gumbel(N_DATA, G, BATCH_SIZE, T)

    # convert one-hot event types to indices; convert normalized timestamps to original
    _gen_seqs_et_ind = tf.argmax(_gen_seqs_et, axis=2).numpy().reshape(N_DATA, T, 1)
    _gen_seqs_ts_ori = apply_mean_std(_gen_seqs_ts, GLOBAL_MEAN, GLOBAL_STD)
    _gen_seqs_for_rules = np.dstack((_gen_seqs_et_ind, _gen_seqs_ts_ori))    

    _mad_score = calculate_MAD_score(_gen_seqs_et_ind)
    _smad_score = calculate_self_MAD_score(_gen_seqs_et_ind)
    _fid_score = calculate_FID_score(_gen_seqs_ts_ori)
    _rule_score = calculate_rule_score(_gen_seqs_for_rules)
    
    return {'mad' : _mad_score, 'fid' : _fid_score, 'oracle' : _rule_score, 'smad' : _smad_score}


def track_training(step, G, save_path=None, verbose=True, plot=True, save_G=True, save_D=True):
    _gen_seqs_et, _gen_seqs_ts = generate_sequences_gumbel(N_DATA, G, BATCH_SIZE, T)
    
    # convert one-hot event types to indices; convert normalized timestamps to original
    _gen_seqs_et_ind = tf.argmax(_gen_seqs_et, axis=2).numpy().reshape(N_DATA, T, 1)
    _gen_seqs_ts_ori = apply_mean_std(_gen_seqs_ts, GLOBAL_MEAN, GLOBAL_STD)
    _gen_seqs_for_rules = np.dstack((_gen_seqs_et_ind, _gen_seqs_ts_ori))
    
    _mad_score = calculate_MAD_score(_gen_seqs_et_ind)
    _fid_score = calculate_FID_score(_gen_seqs_ts_ori)
    _rule_score = calculate_rule_score(_gen_seqs_for_rules)

    if verbose:
        print('event_types:', _gen_seqs_et_ind[0,:, :].squeeze().tolist())
        print('mad_score:', _mad_score)
        print('fid_score:', _fid_score)
        print('rule_score:', _rule_score)

    if plot:
        plt.figure()
        x = np.arange(_gen_seqs_et[0,:,:].shape[0])
        y = _gen_seqs_ts[0,:,:]
        plt.plot(x, y)
        plt.title(f'wave shape after {step} steps')
        plt.show()

    if save_path and save_G:
        G_save_path = os.path.join(save_path, f'G_{step}', 'model_weights.tf')
        G.save_weights(G_save_path)
        print('G saved to:', G_save_path)
        
    if save_path and save_D:
        D_save_path = os.path.join(save_path, f'D_{step}', 'model_weights.tf')
        D.save_weights(D_save_path)
        print('D saved to:', D_save_path)
        
    return _mad_score, _fid_score, _rule_score

def train_generator_gumbel(generator, discriminator, batch_size, T, verbose=False,                   
                optimizer=Adam(lr=0.001), temperature=0.5, use_tlstm=False):
    
    with tf.GradientTape() as tape:                        

        G_sample_et, G_sample_ts = generate_sequences_gumbel(batch_size, generator, batch_size, T, temperature, use_tlstm)
        D_fake = discriminator([G_sample_et, G_sample_ts])

        generator_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(D_fake), D_fake))
        
    if verbose:
        print('generator loss:{}'.format(generator_loss))
        print('-----------------------')

    # update generator
    generator_grads = tape.gradient(generator_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))

    return generator_loss


def train_discriminator_gumbel(real_data_batch_et, real_data_batch_ts, generator, discriminator, batch_size, T, verbose=False,
                        optimizer=Adam(lr=0.001), label_smoothing=False, label_flipping=False):

    # data prep
    real_labels = tf.ones((batch_size//2, 1))        
    fake_labels = tf.zeros((batch_size//2, 1))    
    
    real_data_batch_et, real_data_batch_ts = real_data_batch_et[:batch_size//2, :, :], real_data_batch_ts[:batch_size//2, :, :]    
    fake_data_batch_et, fake_data_batch_ts = generate_sequences_gumbel(batch_size//2, generator, batch_size, T)
    
    if label_smoothing:
        fake_labels = fake_labels + tf.random.normal(fake_labels.shape, mean=0, stddev=0.3)
        fake_labels = tf.clip_by_value(fake_labels, clip_value_min=0., clip_value_max=0.3)

        real_labels = real_labels + tf.random.normal(real_labels.shape, mean=0, stddev=0.3)
        real_labels = tf.clip_by_value(real_labels, clip_value_min=0.7, clip_value_max=1.0)

    if label_flipping:
        if tf.random.uniform((1,)) < 0.05:
            fake_labels, real_labels = real_labels, fake_labels
    
    total_data_et = tf.concat([fake_data_batch_et, real_data_batch_et], axis=0)
    total_data_ts = tf.concat([fake_data_batch_ts, real_data_batch_ts], axis=0)
    total_labels = tf.concat([fake_labels, real_labels], axis=0)        
        
    # train the discriminator
    with tf.GradientTape() as tape:                                                           
        # train discriminator
        pred_prob = discriminator([total_data_et, total_data_ts])

        # cross-entropy loss
        discriminator_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(total_labels, pred_prob, from_logits=False))

        # average true return
        average_true_return = tf.reduce_mean(pred_prob)
        
        if verbose:
            print('total discriminator loss:{}'.format(discriminator_loss))
            print('average true return:{}'.format(average_true_return))
            print('-----------------------')

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return discriminator_loss, average_true_return


def rollout_from_initial_mcc(G, batch_size, init_et, init_ts, L=T, temperature=1, use_tlstm=True):
    # Begin from dummy init state (init_token=1, init_timestamp=0.0)
    all_state_et = init_et
    all_state_ts = init_ts
    all_token_logits = tf.zeros_like(init_et)
    all_time_mu = tf.zeros_like(init_ts)
    all_time_sigma = tf.zeros_like(init_ts)
    
    l_prefix = init_et.shape[1]
    
    G.reset_states()
    
    if use_tlstm:
        # Time-LSTM
        G.layers[4].states[0] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        G.layers[4].states[1] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
    else:
        # LSTM-token
        G.layers[4].states[0] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        G.layers[4].states[1] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        # LSTM-time
        G.layers[5].states[0] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))
        G.layers[5].states[1] = tf.Variable(tf.random.normal(shape=(BATCH_SIZE, HIDDEN_DIM), mean=0.0, stddev=1, dtype=tf.float64))

    for _ in range(L + 1 - l_prefix):  # sequence length
        curr_state_et = all_state_et[:, -1:, :]
        curr_state_ts = all_state_ts[:, -1:, :]

        token_logits, time_mu, time_sigma = G([curr_state_et, curr_state_ts])
        
        # non-differentiable sampling tokens
        sampled_et_ind = tf.random.categorical(token_logits / temperature, num_samples=1, dtype=tf.int32)
        sampled_et = tf.one_hot(sampled_et_ind, depth=EVENT_VOCAB_DIM, axis=2, dtype=tf.float64)
        
        sampled_et = tf.reshape(sampled_et, [batch_size, 1, EVENT_VOCAB_DIM])        
        sampled_et = tf.cast(sampled_et, dtype=tf.float64) # cast sampled_et into float
        
        # non-differentiable sampling time deltas
        time_mu = tf.reshape(time_mu, [batch_size, 1, 1])
        time_sigma = tf.reshape(time_sigma, [batch_size, 1, 1])
        sampled_ts = tf.random.normal((batch_size, 1, 1), mean=time_mu, stddev=time_sigma, dtype=tf.float64)
        
        # reshape for concat
        token_logits = tf.reshape(token_logits, [batch_size, 1, EVENT_VOCAB_DIM])
        
        # Do NOT stop genererating once hit end_token; G is supposed to learn it.
        all_state_et = tf.concat([all_state_et, sampled_et], axis=1)
        all_state_ts = tf.concat([all_state_ts, sampled_ts], axis=1)
        
        all_token_logits = tf.concat([all_token_logits, token_logits], axis=1)
        all_time_mu = tf.concat([all_time_mu, time_mu], axis=1)
        all_time_sigma = tf.concat([all_time_sigma, time_sigma], axis=1)
        
    # the initlal random states are excluded in output
    return all_state_et[:, 1:, :], all_state_ts[:, 1:, :], all_token_logits[:,1:,:], all_time_mu[:,1:,:], all_time_sigma[:,1:,:]


def generate_sequences_mcc(N_gen, generator, batch_size, T, temperature=1, use_tlstm=True):
    """
        Generate sequences batch per batch
    :param N_gen: total number of seqs to be generated
    :param generator:
    :param batch_size:
    :param T:
    :return: a python list of shape [N_gen, T, 1]
    """
    N = 0
    all_type_seq = None
    all_time_seq = None
    all_token_logits = None
    all_time_mu = None
    all_time_sigma = None
    
    init_token_noise, init_time_zeros = generate_initial_noise(zero_time_start=True)
        
    while N < N_gen:
        batch_state_et, batch_state_ts, batch_token_logits, batch_time_mu, batch_time_sigma = rollout_from_initial_mcc(generator, batch_size, init_token_noise, init_time_zeros, T, temperature, use_tlstm)

        if all_type_seq is None or all_time_seq is None:
            all_type_seq = batch_state_et
            all_time_seq = batch_state_ts
            all_token_logits = batch_token_logits
            all_time_mu = batch_time_mu
            all_time_sigma = batch_time_sigma            
        else:
            all_type_seq = tf.concat([all_type_seq, batch_state_et], axis=0)
            all_time_seq = tf.concat([all_time_seq, batch_state_ts], axis=0)
            all_token_logits = tf.concat([all_token_logits, batch_token_logits], axis=0)
            all_time_mu = tf.concat([all_time_mu, batch_time_mu], axis=0)
            all_time_sigma = tf.concat([all_time_sigma, batch_time_sigma], axis=0)

        N += batch_size

    all_type_seq = all_type_seq[:N_gen, :, :]
    all_time_seq = all_time_seq[:N_gen, :, :]
    all_token_logits = all_token_logits[:N_gen, :, :]
    all_time_mu = all_time_mu[:N_gen, :, :]
    all_time_sigma = all_time_sigma[:N_gen, :, :]

    return all_type_seq, all_time_seq, all_token_logits, all_time_mu, all_time_sigma


def train_generator_mcc(generator, discriminator, critic, batch_size, T, 
                        beta_token=1, beta_time=1, verbose=False, optimizer=Adam(lr=0.001),
                        use_advantage=False, temperature=1, use_tlstm=True):
    
    # clear critic states for a new batch
    critic.reset_states()
    
    with tf.GradientTape(persistent=True) as tape:     
                        
        gen_step_loss = []
        critic_step_loss = []

        G_sample_et, G_sample_ts, G_token_logits, G_time_mu, G_time_sigma = generate_sequences_mcc(batch_size, generator, batch_size, T, temperature, use_tlstm)
        
        true_return = discriminator([G_sample_et, G_sample_ts])
        
        # Monte-Carlo with Critic
        for i in range(T):  
            curr_state_et = G_sample_et[:, i:i+1, :]
            curr_state_ts = G_sample_ts[:, i:i+1, :]
            token_logits = G_token_logits[:, i, :]
            time_mu = G_time_mu[:, i, :]
            time_sigma = G_time_sigma[:, i, :]
            
            q_value = critic([curr_state_et, curr_state_ts])
            advantage = true_return - q_value
        
            # averge loss over batch at each rollout step: -E[log_prob * A]
            
            # Token
            p_token_all = tf.nn.softmax(token_logits)
            assert(p_token_all.shape == (batch_size, EVENT_VOCAB_DIM))
            
            chosen_ind = tf.argmax(curr_state_et, axis=2)
            assert(chosen_ind.shape == (batch_size, 1))
                                   
            p_token_chosen = tf.gather_nd(p_token_all, chosen_ind, batch_dims=1) # gather in vocab dim
            p_token_chosen = tf.reshape(p_token_chosen, (batch_size, 1))
            assert(p_token_chosen.shape == (batch_size, 1))
                                   
            token_entropy = -p_token_chosen * tf.math.log(p_token_chosen)
            assert(token_entropy.shape == (batch_size, 1))
            
            if use_advantage:
                token_policy_gradient_loss = -tf.reduce_mean(tf.math.log(p_token_chosen) * (advantage + beta_token * token_entropy))
            else:
                token_policy_gradient_loss = -tf.reduce_mean(tf.math.log(p_token_chosen) * (q_value + beta_token * token_entropy))
            
            # Time
            time_dist = tfp.distributions.Normal(time_mu, time_sigma)
            time_gaussian_log = time_dist.log_prob(tf.reshape(curr_state_ts, (batch_size, 1)))
            assert(time_gaussian_log.shape == (batch_size, 1))
            
            time_entropy = 0.5 * tf.math.log(2*np.pi*np.e*tf.math.square(time_sigma))
            assert(time_entropy.shape == (batch_size, 1))
            
            if use_advantage:
                time_policy_gradient_loss = -tf.reduce_mean(time_gaussian_log * (advantage + beta_time * time_entropy))
            else:
                time_policy_gradient_loss = -tf.reduce_mean(time_gaussian_log * (q_value + beta_time * time_entropy))
            
            gen_step_loss.append(token_policy_gradient_loss + time_policy_gradient_loss)
                        
            critic_mse_loss = tf.reduce_mean(tf.keras.losses.MSE(true_return, q_value))
            critic_step_loss.append(critic_mse_loss)
            
        generator_loss = tf.reduce_mean(gen_step_loss)
        critic_loss = tf.reduce_mean(critic_step_loss)
        
    if verbose:
        print('generator loss:{}'.format(generator_loss))
        print('last advantage:{}'.format(advantage[0,:]))
        print('last log token-prob:{}'.format(tf.math.log(p_token_chosen[0,:])))
        print('last time gaussian:{}'.format(time_gaussian_log[0,:]))
        print('last token entropy:{}'.format(token_entropy[0,:]))
        print('last time entropy:{}'.format(time_entropy[0,:]))
        print('critic loss:{}'.format(critic_loss))
        print('-----------------------')

    # update generator
    generator_grads = tape.gradient(generator_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    
    # update critic
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # explicitly drop tape because persistent=True
    del tape

    return generator_loss, critic_loss


def train_discriminator_mcc(real_data_batch_et, real_data_batch_ts, generator, discriminator, batch_size, T, verbose=False,
                        optimizer=Adam(lr=0.001), label_smoothing=False, label_flipping=False):

    # data prep
    real_labels = tf.ones((batch_size//2, 1))
    fake_labels = tf.zeros((batch_size//2, 1))
    
    real_data_batch_et, real_data_batch_ts = real_data_batch_et[:batch_size//2, :, :], real_data_batch_ts[:batch_size//2, :, :]    
    fake_data_batch_et, fake_data_batch_ts, _, _, _ = generate_sequences_mcc(batch_size//2, generator, batch_size, T)
    
    if label_smoothing:
        fake_labels = fake_labels + tf.random.normal(fake_labels.shape, mean=0, stddev=0.3)
        fake_labels = tf.clip_by_value(fake_labels, clip_value_min=0., clip_value_max=0.3)

        real_labels = real_labels + tf.random.normal(real_labels.shape, mean=0, stddev=0.3)
        real_labels = tf.clip_by_value(real_labels, clip_value_min=0.7, clip_value_max=1.0)

    if label_flipping:
        if tf.random.uniform((1,)) < 0.05:
            fake_labels, real_labels = real_labels, fake_labels
    
    total_data_et = tf.concat([fake_data_batch_et, real_data_batch_et], axis=0)
    total_data_ts = tf.concat([fake_data_batch_ts, real_data_batch_ts], axis=0)
    total_labels = tf.concat([fake_labels, real_labels], axis=0)        
        
    # train the discriminator
    with tf.GradientTape() as tape:                                                           
        # train discriminator
        pred_prob = discriminator([total_data_et, total_data_ts])

        # cross-entropy loss
        discriminator_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(total_labels, pred_prob, from_logits=False))

        # average true return
        average_true_return = tf.reduce_mean(pred_prob)
        
        if verbose:
            print('total discriminator loss:{}'.format(discriminator_loss))
            print('average true return:{}'.format(average_true_return))
            print('-----------------------')

    grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return discriminator_loss, average_true_return