import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, LSTM, Embedding, Reshape, Dense, Dropout
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import regularizers

from sgtlstm.TimeLSTM import TimeLSTM0, TimeLSTM1, TimeLSTM2, TimeLSTM3

tf.keras.backend.set_floatx('float64')


def build_D(T, event_vocab_dim, emb_dim, hidden_dim=11):
    """
        Build a discriminator for event type sequence of shape (batch_size, T, input_dim)
        and input event type sequence of shape (batch_size, T, 1)
    :param T: length of the sequence
    :param event_vocab_dim: size of event vocabulary ['na', 'init', 'start', 'view', 'click', 'install']
    :param emb_dim: dimension of the embedding layer output for event type
    :param hidden_dim: dimension hidden of the time lstm cell
    :return: discriminator D
    """
    # Time-LSTM:
    i_et = Input(shape=(T, 1), name='event_type')  # input of discrete feature event type
    i_ts = Input(shape=(T, 1), name='time_delta')  # input of continuous feature timestamp
    mask_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(T, 1))
    masked_ts = mask_layer(i_ts)
    masked_et = mask_layer(i_et)

    embed0 = Embedding(input_dim=event_vocab_dim, output_dim=emb_dim, input_length=T, mask_zero=True)(masked_et)
    embed0 = Reshape((T, emb_dim))(embed0)  # shape=[Batch_size, T, emb_dim]
    merged0 = tf.keras.layers.concatenate([embed0, masked_ts], axis=2)  # # shape=[Batch_size, T, emb_dim + time_dim]

    hm, tm = TimeLSTM1(hidden_dim, activation='selu', name='time_lstm', return_sequences=False)(merged0)

    time_comb = tf.keras.layers.concatenate([hm, tm], axis=1)

    # predicted real prob
    real_prob = Dense(1, activation='sigmoid', name='fraud_prob', regularizers=regularizers.l1_l2(l1=1e-3, l2=1e-3))(
        time_comb)

    discriminator = Model(
        inputs=[i_et, i_ts],
        outputs=[real_prob])

    return discriminator


def build_G(batch_size, event_vocab_dim, emb_dim, hidden_dim=11):
    """
        Build a generator for event type sequence of shape (batch_size, T, input_dim)
        and input event type sequence of shape (batch_size, T, 1)
    :param batch_size: batch size must been specified at generator
    :param event_vocab_dim: size of event vocabulary ['na', 'start', 'click', 'install']
    :param emb_dim: dimension of the embedding layer output for event type
    :param hidden_dim: dimension hidden of the time lstm cell
    :return:
    """
    # Time-LSTM:
    i_et = Input(batch_shape=(batch_size, None, 1), name='event_type')  # input of discrete feature event type
    i_ts = Input(batch_shape=(batch_size, None, 1), name='time_delta')  # input of continuous feature timestamp

    mask_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(None, 1))
    masked_et = mask_layer(i_et)
    masked_ts = mask_layer(i_ts)

    embed0 = Embedding(input_dim=event_vocab_dim, output_dim=emb_dim, mask_zero=True)(masked_et)
    embed0 = Reshape([1, emb_dim])(embed0)
    merged0 = tf.keras.layers.concatenate([embed0, masked_ts], axis=2)

    hm, tm = TimeLSTM1(hidden_dim, activation='selu', name='time_lstm',
                       stateful=True, return_sequences=False)(merged0)
    # time_comb = tf.keras.layers.concatenate([hm, tm], axis=1)
    time_out = Dense(1 + 1, activation='linear', name='output')(tm)
    time_out = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1 + tf.math.softplus(t[..., 1:])),
        name='Normal')(time_out)

    # predicted prob of next token
    token_prob = Dense(event_vocab_dim, activation='softmax', name='token_prob')(hm)
    generator = Model(
        inputs=[i_et, i_ts],
        outputs=[token_prob, time_out])
    return generator


def build_D_2(T, event_vocab_dim, emb_dim, hidden_dim=11):
    """
        Build a discriminator for event type sequence of shape (batch_size, T, input_dim)
        and input event type sequence of shape (batch_size, T, 1)
    :param T: length of the sequence
    :param event_vocab_dim: size of event vocabulary ['na', 'init', 'start', 'view', 'click', 'install']
    :param emb_dim: dimension of the embedding layer output for event type
    :param hidden_dim: dimension hidden of the time lstm cell
    :return: discriminator D
    """
    # Time-LSTM:
    i_et = Input(shape=(T, 1), name='event_type')  # input of discrete feature event type
    i_ts = Input(shape=(T, 1), name='time_delta')  # input of continuous feature timestamp
    mask_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(T, 1))
    masked_ts = mask_layer(i_ts)
    masked_et = mask_layer(i_et)

    embed0 = Embedding(input_dim=event_vocab_dim, output_dim=emb_dim, input_length=T, mask_zero=True)(masked_et)
    embed0 = Reshape((T, emb_dim))(embed0)  # shape=[Batch_size, T, emb_dim]
    merged0 = tf.keras.layers.concatenate([embed0, masked_ts], axis=2)  # # shape=[Batch_size, T, emb_dim + time_dim]

    hm, tm = TimeLSTM1(hidden_dim, activation='selu', name='time_lstm', return_sequences=False)(merged0)

    time_comb = tf.keras.layers.concatenate([hm, tm], axis=1)

    # predicted real prob
    real_prob = Dense(1, activation='sigmoid', name='fraud_prob',
                      kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))(
        time_comb)

    discriminator = Model(
        inputs=[i_et, i_ts],
        outputs=[real_prob])

    return discriminator


def build_G_2(batch_size, event_vocab_dim, emb_dim, hidden_dim=11):
    """
        Build a generator for event type sequence of shape (batch_size, T, input_dim)
        and input event type sequence of shape (batch_size, T, 1)
    :param batch_size: batch size must been specified at generator
    :param event_vocab_dim: size of event vocabulary ['na', 'start', 'click', 'install']
    :param emb_dim: dimension of the embedding layer output for event type
    :param hidden_dim: dimension hidden of the time lstm cell
    :return:
    """
    # Time-LSTM:
    i_et = Input(batch_shape=(batch_size, None, 1), name='event_type')  # input of discrete feature event type
    i_ts = Input(batch_shape=(batch_size, None, 1), name='time_delta')  # input of continuous feature timestamp

    mask_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(None, 1))
    masked_et = mask_layer(i_et)
    masked_ts = mask_layer(i_ts)

    embed0 = Embedding(input_dim=event_vocab_dim, output_dim=emb_dim, mask_zero=True)(masked_et)
    embed0 = Reshape([1, emb_dim])(embed0)
    merged0 = tf.keras.layers.concatenate([embed0, masked_ts], axis=2)

    hm, tm = TimeLSTM1(hidden_dim, activation='selu', name='time_lstm',
                       stateful=True, return_sequences=False)(merged0)

    time_comb = tf.keras.layers.concatenate([hm, tm], axis=1)

    dense_time = Dense(hidden_dim // 2, activation='linear', name='dense_time')(time_comb)
    dense_token = Dropout(rate=0.4)(dense_time)
    time_out = Dense(1 + 1, activation='linear', name='output')(dense_token)
    time_out = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1 + tf.math.softplus(t[..., 1:])),
        name='Normal')(time_out)

    # predicted prob of next token
    dense_token = Dense(hidden_dim // 2, activation='linear', name='dense_token')(time_comb)
    dense_token = Dropout(rate=0.4)(dense_token)
    token_prob = Dense(event_vocab_dim, activation='softmax', name='token_prob')(dense_token)
    generator = Model(
        inputs=[i_et, i_ts],
        outputs=[token_prob, time_out])
    return generator


def build_critic(batch_size, event_vocab_dim, emb_dim, hidden_dim=11):
    """
        Build a critic network for event type sequence of shape (batch_size, T, input_dim)
        and input event type sequence of shape (batch_size, T, 1)
    :param batch_size: size of batch
    :param event_vocab_dim: size of event vocabulary ['na', 'start', 'click', 'install']
    :param emb_dim: dimension of the embedding layer output for event type
    :param hidden_dim: dimension hidden of the time lstm cell
    :return: model_critic
    """
    # Time-LSTM:
    i_et = Input(batch_shape=(batch_size, None, 1), name='event_type')  # input of discrete feature event type
    i_ts = Input(batch_shape=(batch_size, None, 1), name='time_delta')  # input of continuous feature timestamp

    mask_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(None, 1))
    masked_et = mask_layer(i_et)
    masked_ts = mask_layer(i_ts)
    embed0 = Embedding(input_dim=event_vocab_dim, output_dim=emb_dim, mask_zero=True)(masked_et)
    embed0 = Reshape([1, emb_dim])(embed0)
    merged0 = tf.keras.layers.concatenate([embed0, masked_ts], axis=2)

    hm, tm = TimeLSTM1(hidden_dim, activation='selu', name='time_lstm',
                       stateful=True, return_sequences=False)(merged0)
    time_comb = tf.concat([hm, tm], axis=1)

    # critic value for future rewards
    critic_value = Dense(1, activation='sigmoid', name='critic_value')(time_comb)

    model_critic = Model(
        inputs=[i_et, i_ts],
        outputs=[critic_value]
    )

    return model_critic
