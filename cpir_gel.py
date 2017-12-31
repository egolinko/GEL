import numpy as np
import pandas as pd
import itertools
from functools import partial
from scipy.linalg import block_diag

def five_bin(x):
    if len(x.unique()) < 5:
        ret = x
    elif x.dtype == "int64":
        ret = pd.cut(np.array(x), 5, labels=False)
    else:
        ret = x
    return x.astype('str')


def get_diag_index(d_, l):
    idx = d_[d_.Class == d_.Class.value_counts().index[l]].index
    return idx


def row_feature_rep(rows_, features_):
    r_1 = rows_.mean(axis=1).as_matrix()
    f_1 = features_.mean(axis=0).as_matrix()

    r_0 = 1 - r_1
    f_0 = 1 - f_1

    f = np.array([f_0, f_1])
    r = np.array([r_0, r_1])

    Q_ = np.matmul(f.transpose(), r)

    return Q_


def get_upper_Fs(d_, ccm, i):
    ret = row_feature_rep(rows_= d_[d_.Class == ccm.cj[i]]
                          .drop("Class", axis=1),
                          features_=d_[(d_.Class == ccm.ci[i]) | (d_.Class == ccm.cj[i])]
                          .drop("Class", axis=1))
    return ret


def get_lower_Fs(d_, ccm, i):
    ret = row_feature_rep(rows_ = d_[d_.Class == ccm.ci[i]]
                          .drop("Class", axis=1),
                          features_ = d_[(d_.Class == ccm.ci[i]) | (d_.Class == ccm.cj[i])]
                          .drop("Class", axis=1))
    return ret


def get_diag(d_, diag_idx_, i):
    ret = row_feature_rep(rows_ = d_.iloc[diag_idx_[i]]
                          .drop("Class", axis=1),
                          features_ = d_.iloc[diag_idx_[i]]
                          .drop("Class", axis=1))
    return ret


def makeMat(k_, which_diag, ccm, d_, Fs, D_):
    if which_diag == 'upper':
        l = ccm[ccm.ci == d_.Class.value_counts().index[k_]].index
    else:
        l = ccm[ccm.cj == d_.Class.value_counts().index[k_]].index

    if (len(l) == 0 and which_diag == 'upper'):
        Q_ = D_[len(d_.Class.unique()) - 1]
    elif (len(l) == 0 and which_diag == 'lower'):
        Q_ = D_[0]
    else:
        q = [Fs[l] for l in l.values]
        q_ = np.concatenate(q, axis=1)
        if (which_diag == 'upper'):
            Q_ = np.concatenate([D_[k_], q_], axis=1)
        else:
            Q_ = np.concatenate([q_, D_[k_]], axis=1)

    if (Q_.shape[1] == d_.shape[0]):
        Q = Q_
    else:
        if (which_diag == 'upper'):
            Q = np.concatenate([np.zeros((d_.shape[1] - 1, d_.shape[0] - Q_.shape[1])),
                                Q_], axis=1)
        else:
            Q = np.concatenate([Q_,
                                np.zeros((d_.shape[1] - 1, d_.shape[0] - Q_.shape[1]))],
                               axis=1)
    return Q


def cpir_gel(source_data_, k, learning_method):
    '''

    Args:
        source_data_: a pandas dataframe with the class label named 'Class'
        k: number of eigenvectors to use for new embedding, if  'max' dim(source_data_) = dim(emb)
        learning_method: 'unsupervised' indicates no class label, otherwise 'supervised'

    Returns:
        emb: new embedded space
        mb: one-hot data
        source_data_: original data frame
    '''

    if learning_method == 'supervised':
        m = source_data_.drop("Class", axis=1).apply(five_bin, axis=1)
        mb_ = pd.get_dummies(m)
        mb_['Class'] = pd.Categorical(source_data_.Class,
                                      categories=source_data_.Class.value_counts()
                                      .keys()
                                      .tolist(),
                                      ordered=True)
        mb = mb_.sort_values(by='Class')

        class_combs = pd.DataFrame(
            list(itertools.combinations(
                mb.Class.value_counts().index, 2)),
            columns=['ci', 'cj']
        )

        diag_idx = map(partial(get_diag_index, mb), range(len(mb.Class.unique())))

        D = map(partial(get_diag, mb, diag_idx), range(len(diag_idx)))
        upper_Fs = map(partial(get_upper_Fs, mb, class_combs), range(len(class_combs)))
        lower_Fs = map(partial(get_lower_Fs, mb, class_combs), range(len(class_combs)))

        upper_block = np.concatenate(
            map(partial(makeMat, which_diag="upper",
                        ccm=class_combs,
                        d_=mb, Fs=upper_Fs, D_=D),
                range(len(diag_idx))))

        lower_block = np.concatenate(
            map(partial(makeMat, which_diag="lower",
                        ccm=class_combs,
                        d_=mb, Fs=lower_Fs, D_=D),
                range(len(diag_idx))))

        b = block_diag(*map(lambda x: np.full(D[x].shape, .5),
                            range(len(D))))
        b[b == 0] = 1

        A = block_diag(*map(lambda x: np.full((mb.Class.value_counts().values[x],
                                               mb.Class.value_counts().values[x]),
                                              mb.Class.value_counts(normalize=True).values[x]),
                            range(len(D)))
                       )

        Q_ = (upper_block + lower_block) * b
        Q = np.matmul(Q_.transpose(), Q_)
        S_ = np.matmul(np.divide(Q, np.max(Q)) * A, mb.drop("Class", axis=1).as_matrix())
        U, s, V = np.linalg.svd(S_)

    else:
        m = source_data_.apply(five_bin, axis=1)
        mb = pd.get_dummies(m)

        u = row_feature_rep(rows_= mb, features_= mb)
        Q = np.matmul(u.transpose(), u)
        S_ = np.matmul(np.divide(Q, np.max(Q)), mb.as_matrix())
        U, s, V = np.linalg.svd(S_)

    if k == 'max':
        v_t = V.transpose()
    else:
        v_t = V.transpose()[:, 0:k]

    if learning_method == 'supervised':
        emb = np.matmul(mb.drop("Class", axis=1).as_matrix(), v_t)
    else:
        emb = np.matmul(mb.as_matrix(), v_t)

    return emb, v_t, mb, source_data_
