import os
import sys
import json

from collections import Counter, defaultdict
import numpy as np
from collections import Counter
from itertools import combinations

from scipy.linalg import sqrtm
import tensorflow as tf

# Rule Based Quality

# 0 is reserved for padding 1 is for 'init token'
EVENT_TYPES = ['P', 'N', 'A', 'B', 'C', 'D']
EVENT_ENCODE = {'P': 0, 'N': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5}
INIT_TOKEN = EVENT_ENCODE['N']
END_TOKEN = EVENT_ENCODE['P']

MIN_SAME_DELAY = 10
MAX_PAIR_DELAY = 50


def check_increasing_rule(seq):
    for i in range(1, len(seq)):
        if seq[i][1] <= seq[i - 1][1]:
            return False
    return True


def check_rule_1(seq, use_init_token=True):
    if use_init_token:
        seq = seq[1:]
    return seq[0][0] == EVENT_ENCODE['A']


def check_rule_2(seq, use_init_token=True):
    if use_init_token:
        seq = seq[1:]
    cnt = Counter()
    for et, dt in seq:
        cnt[et] += 1
    # rule 2
    if len(cnt.keys()) > 3 and EVENT_ENCODE['A'] in cnt.keys():
        return True
    else:
        return False


def check_rule_3(seq, use_init_token=True):
    if use_init_token:
        seq = seq[1:]
        # one-pass: add D to queue to be attributed to the first available C in a reversed linear scanning
    queue = []
    for i in range(len(seq) - 1, -1, -1):
        if seq[i][0] == EVENT_ENCODE['D']:  # encounter a D event
            queue.append(i)
        elif seq[i][0] == EVENT_ENCODE['C'] and queue:  # encounter a C event
            queue.pop(0)
    return len(queue) == 0


def check_rule_4(seq, use_init_token=True):
    if use_init_token:
        seq = seq[1:]
    cnt = Counter()
    for et, dt in seq:
        cnt[et] += 1
    # rule 4
    if cnt[EVENT_ENCODE['A']] < EVENT_ENCODE['B']:
        return False
    if cnt[EVENT_ENCODE['B']] < EVENT_ENCODE['C']:
        return False
    if cnt[EVENT_ENCODE['C']] < EVENT_ENCODE['D']:
        return False
    return True


def check_rule_5(seq, use_init_token=True):
    if use_init_token:
        seq = seq[1:]
    prev_et, _ = EVENT_ENCODE['N'], 0.0
    for et, dt in seq:
        if et == prev_et and dt < MIN_SAME_DELAY:
            return False
        prev_et = et
    return True


def check_rule_6(seq, use_init_token=True):
    if use_init_token:
        seq = seq[1:]

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

    ets = [e[0] for e in seq]
    tss = recover_timedelta_to_timestamp([e[1] for e in seq])

    # one-pass: add D to queue to be attributed to the first available C in a reversed linear scanning
    queue = []
    for i in range(len(seq) - 1, -1, -1):
        if ets[i] == EVENT_ENCODE['D']:  # encounter a D event
            queue.append(i)
        elif ets[i] == EVENT_ENCODE['C'] and queue:  # encounter a C event
            if tss[queue[0]] - tss[i] <= MAX_PAIR_DELAY:
                queue.pop(0)
            else:
                return False
    # for rule 6, it's fine if there are unpaired D in queue
    # b/c this rules is to ensure for each paired (C, D), the delay is bounded
    return True


def get_rule_dist(seqs, use_init_token=True):
    seq_to_rules = [0] * 7
    N = len(seqs)

    for i in range(N):
        seq = seqs[i]
        # check rules one by one:
        if check_rule_1(seq):
            seq_to_rules[1] += 1
        if check_rule_2(seq):
            seq_to_rules[2] += 1
        if check_rule_3(seq):
            seq_to_rules[3] += 1
        if check_rule_4(seq):
            seq_to_rules[4] += 1
        if check_rule_5(seq):
            seq_to_rules[5] += 1
        if check_rule_6(seq):
            seq_to_rules[6] += 1

    return [freq / N for freq in seq_to_rules[1:]]


def get_rule_foreach(seqs, use_init_token=True):
    seq_to_rules = defaultdict(set)
    N = len(seqs)

    for i in range(N):
        seq = seqs[i]
        # check rules one by one:
        if check_rule_1(seq):
            seq_to_rules[i].add(1)
        if check_rule_2(seq):
            seq_to_rules[i].add(2)
        if check_rule_3(seq):
            seq_to_rules[i].add(3)
        if check_rule_4(seq):
            seq_to_rules[i].add(4)
        if check_rule_5(seq):
            seq_to_rules[i].add(5)
        if check_rule_6(seq):
            seq_to_rules[i].add(6)

    return seq_to_rules


def get_all_combs(rules=[1, 2, 3, 4, 5, 6]):
    all_combs = set()
    for k in range(1, len(rules) + 1):
        all_combs.update(set(combinations(rules, k)))
    return all_combs


def check_all_combinations(rule_dict):
    rules = [1, 2, 3, 4, 5, 6]
    combs = set()
    comb_dict = defaultdict(Counter)

    for k in range(1, len(rules) + 1):
        combs.update(set(combinations(rules, k)))

    for i, rule_list in rule_dict.items():
        for c in combs:
            if set(c).issubset(set(rule_list)):
                comb_dict[i][c] += 1

    return comb_dict


def get_comb_dist(all_combs, comb_dict):
    comb_dist = {c: 0 for c in all_combs}
    comb_dist.update({(0,): 0})  # no rules followed

    for d in comb_dict.values():
        if not d or len(d) == 0:
            comb_dist[(0,)] += 1
            continue
        for c in all_combs:
            if c in d:
                comb_dist[c] += 1

    return comb_dist


def seqs_to_comb_dist(seqs, all_combs):
    seq_to_rules = get_rule_foreach(seqs, True)
    comb_results = check_all_combinations(seq_to_rules)

    comb_freq = get_comb_dist(all_combs, comb_results)

    # sort the combo distribution
    all_comb_list = [(0,)] + sorted(list(all_combs), key=lambda e: (len(e), e))

    sorted_comb_dist = [(comb, comb_freq[comb]) for comb in all_comb_list]

    return comb_freq, sorted_comb_dist


def seqs_to_scores(seqs):
    seq_to_rules = get_rule_foreach(seqs, True)
    comb_results = check_all_combinations(seq_to_rules)
    seq_to_score = dict()

    for i, combs in comb_results.items():
        score = 0
        for comb, cnt in combs.items():
            if comb == (0,):
                continue
            score += 2 ** len(comb) * cnt
        seq_to_score[i] = score

    return sum(seq_to_score.values()) / len(seq_to_score)


def calculate_FID_batch(batch1, batch2, weight=1):
    mu1, sigma1 = batch1.mean(axis=0), np.cov(batch1, rowvar=False)
    mu2, sigma2 = batch2.mean(axis=0), np.cov(batch2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid2 = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid2


class MAD:
    """ Sum square of one-hot distance between tokens
    Params:
    -- batch1         : The first batch of sequences to be compared with the second one
                        or it can be the comparison base
    -- batch2
    Returns:
    --   : dict, sum square of distances and base medians
    """

    def __init__(self):
        self.base_ssad = []
        self.base_med = []
        self.base_med_oh = []
        self.one_hot_map = {0: [0, 0, 0, 0, 0],
                            1: [1, 0, 0, 0, 0],
                            2: [0, 1, 0, 0, 0],
                            3: [0, 0, 1, 0, 0],
                            4: [0, 0, 0, 1, 0],
                            5: [0, 0, 0, 0, 1]}

    def nan_if(self, arr, value):
        return np.where(arr == value, np.nan, arr)

    def fit(self, batch1):
        assert len(batch1) > 1, 'batch1 should have more than one sequence'
        X_tok = []
        for i in batch1:
            X_tok.append([j[0] for j in i])
        X_tok_t = np.array(X_tok).T
        X_tok_oh = [[self.one_hot_map[j] for j in i] for i in X_tok]
        medians = [int(i) for i in np.nanmedian(self.nan_if(X_tok_t, 0), axis=1)]
        medians_oh = [self.one_hot_map[j] for j in medians]
        self.base_med = medians
        self.base_med_oh = medians_oh
        tok_dist = []
        for tok in X_tok_oh:
            dist = []
            for j in range(len(tok)):
                tt = np.array(tok[j])
                med = np.array(medians_oh[j])
                # skip 0
                dist.append(np.nansum(np.abs(self.nan_if(tt, 0) - med)))
            tok_dist.append(dist)
        self.base_ssad = np.mean(tok_dist)

    def compare(self, batch2):
        assert len(batch2) > 1, 'batch2 should have more than one sequence'
        X_tok = []
        for i in batch2:
            X_tok.append([j[0] for j in i])
        X_tok_t = np.array(X_tok).T
        X_tok_oh = [[self.one_hot_map[j] for j in i] for i in X_tok]
        medians = [int(i) for i in np.nanmedian(self.nan_if(X_tok_t, 0), axis=1)]
        medians_oh = [self.one_hot_map[j] for j in medians]
        tok_dist = []
        for tok in X_tok_oh:
            dist = []
            for j in range(len(tok)):
                tt = np.array(tok[j])
                med = np.array(self.base_med_oh[j])
                # skip 0
                dist.append(np.nansum(np.abs(self.nan_if(tt, 0) - med)))
            tok_dist.append(dist)
        compare_ssd = np.mean(tok_dist)
        return {'mad': compare_ssd,
                'base_medians': np.array(self.base_med),
                'base_medians_oh': np.array(self.base_med_oh),
                'comp_medians': np.array(medians),
                'comp_medians_oh': np.array(medians_oh)}


def rbf_mmd2(X, Y, sigma=1, biased=True):
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    """
    """
    if isinstance(sigmas, list):
        sigmas = tf.convert_to_tensor(np.array(sigmas))
    if wts is None:
        # print('sigmas:{}'.format(sigmas))
        wts = [1.0] * sigmas.get_shape()[0]

    # debug!
    if len(X.shape) == 2:
        # matrix
        XX = tf.matmul(X, X, transpose_b=True)
        XY = tf.matmul(X, Y, transpose_b=True)
        YY = tf.matmul(Y, Y, transpose_b=True)
    elif len(X.shape) == 3:
        # tensor -- this is computing the Frobenius norm
        XX = tf.tensordot(X, X, axes=[[1, 2], [1, 2]])
        XY = tf.tensordot(X, Y, axes=[[1, 2], [1, 2]])
        YY = tf.tensordot(Y, Y, axes=[[1, 2], [1, 2]])
    else:
        raise ValueError(X)

    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(tf.unstack(sigmas, axis=0), wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float64)
    n = tf.cast(K_YY.get_shape()[0], tf.float64)

    if biased:
        c1 = tf.reduce_sum(K_XX) / (m * m)
        c2 = tf.reduce_sum(K_YY) / (n * n)
        c3 = 2 * tf.reduce_sum(K_XY) / (m * n)
        mmd2 = c1 + c2 - c3
        # mmd2 = (tf.reduce_sum(K_XX) / (m * m)
        #       + tf.reduce_sum(K_YY) / (n * n)
        #       - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
                + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
                - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def get_G_metrics(pos_batch, gen_batch):
    rbq = seqs_to_scores(gen_batch)
    fid = calculate_FID_batch(pos_batch[:, :, 1], gen_batch[:, :, 1])

    mad_obj = MAD()
    mad_obj.fit(pos_batch)
    mad = mad_obj.compare(gen_batch)['mad']

    mmd = rbf_mmd2(pos_batch, gen_batch).numpy()
    mmd_et = rbf_mmd2(pos_batch[:, :, 0], gen_batch[:, :, 0]).numpy()
    mmd_ts = rbf_mmd2(pos_batch[:, :, 1], gen_batch[:, :, 1]).numpy()

    return [rbq, fid, mad, mmd, mmd_et, mmd_ts]


def get_hidden_metrics(time_comb_1, time_comb_2):
    fid = calculate_FID_batch(time_comb_1.numpy(), time_comb_2.numpy())
    mmd = rbf_mmd2(time_comb_1, time_comb_2).numpy()

    return [fid, mmd]