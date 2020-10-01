import pickle
import numpy as np
import tensorflow as tf


def load_fixed_length_sequence_from_pickle(pickle_file_path, to_timedelta=True, end_token=0):
    """
        A list of sequence in format of (event_type, timestamp)
        [[(1, 11), (1, 24), (2, 37), (3, 47), (2, 63), (2, 80), (1, 88), (2, 95), (2, 104), (3, 111)], ...]
    :param pickle_file_path: e.g. /.../project-basileus/seq-gan/data/fixed_length/valid_sequences.pickle
    :param to_timedelta: if True, convert absolute time to timedelta
    :param end_token: same as padding token. default 0
    :return: (event_type_seqs, timestamp_seqs)
    """
    with open(pickle_file_path, 'rb') as f:
        raw_seqs = pickle.load(f)

    if not raw_seqs or not raw_seqs[0]:
        return np.array([]), np.array([])

    N = len(raw_seqs)
    T = len(raw_seqs[0])

    event_type_seqs = []
    timestamp_seqs = []

    for seq in raw_seqs:
        _ets, _tss = [], []
        ts_prev = 0
        for et, ts in seq:
            if et == end_token:
                raise ValueError('Actual data should NOT contain END_TOKEN', end_token)
            _ets.append(et)  # 0 is for padding, standing for END_TOKEN or 'N/A'
            if to_timedelta:
                _tss.append(ts - ts_prev)
                ts_prev = ts
            else:
                _tss.append(ts)
        event_type_seqs.append(_ets)
        timestamp_seqs.append(_tss)

    event_type_seqs = np.array(event_type_seqs).astype(np.float64).reshape((N, T, 1))
    timestamp_seqs = np.array(timestamp_seqs).astype(np.float64).reshape((N, T, 1))

    return event_type_seqs, timestamp_seqs


def create_dataset(features: np.array, labels: np.array, batch_size=2, epochs=10, buffer_size=10000):
    """
    Create dataset from numpy arrays
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def recover_timedelta_to_timestamp(time_seq):
    csum = []
    curr = 0

    for dt in time_seq:
        if dt != 0:
            curr += dt
            csum.append(curr)
        else:
            csum.append(0)

    return csum
