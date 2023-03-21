import numpy as np
my_seed = 666
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os.path as osp

import forest2vec as fv
import utils
from sklearn import metrics
import multiprocessing as mp
from itertools import product

def run(n_samples,
        n_estimators, max_features, bootstrap,
        min_samples_leaf, repeat, vec_dim,
        times, mini_batch, batch_size, lr, decay):
    des_dir = "results/circle"
    X_all, y_all = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.04, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=0, stratify=y_all)

    colors = np.asarray(["red", "blue"])
    save_path = osp.join(des_dir, "input.pdf")
    utils.plot2d(X_test, color=colors[y_test], save_path=save_path)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap,
                                 n_jobs=-1, random_state=0).fit(X_train, y_train)

    drlf = fv.DRLForest(timer=True, print_loss=True)
    X_train_embedding = drlf.fit_transform(X_train, y_train, rfc, min_samples_leaf=min_samples_leaf,
                                           vec_dim=vec_dim, times=times, mini_batch=mini_batch,
                                           batch_size=batch_size, lr=lr, decay=decay, repeat=repeat)
    accuracy_train, SC_train_old, SC_train_new, CH_train_old, CH_train_new, DB_train_old, DB_train_new\
        = utils.get_score(X_train, X_train_embedding, y_train, rfc)

    X_test_embedding = drlf.transform(X_test, X_train_embedding, rfc)
    save_path = osp.join(des_dir, "output.pdf")
    utils.plot3d(X_train_embedding, color=colors[y_train], save_path=save_path)
    accuracy_test, SC_test_old, SC_test_new, CH_test_old, CH_test_new, DB_test_old, DB_test_new \
        = utils.get_score(X_test, X_test_embedding, y_test, rfc)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='liblinear', random_state=0).fit(X_train_embedding, y_train)
    acc_lr_train = clf.score(X_train_embedding, y_train)
    acc_lr_test = clf.score(X_test_embedding, y_test)

    results = pd.DataFrame([[rfc.n_estimators, rfc.max_features, rfc.bootstrap,
                             min_samples_leaf, repeat, vec_dim,
                             drlf.uv_len, drlf.times, mini_batch, drlf.batch_size, lr, decay,
                             accuracy_train, acc_lr_train, SC_train_old, SC_train_new,
                             CH_train_old, CH_train_new, DB_train_old, DB_train_new,
                             accuracy_test, acc_lr_test, SC_test_old, SC_test_new,
                             CH_test_old, CH_test_new, DB_test_old, DB_test_new]])
    save_path = osp.join(des_dir, "results.csv")
    results.to_csv(save_path, mode='a', encoding='utf-8', index=False, header=False)


if __name__ == "__main__":
    n_samples = 10000
    n_estimators = 100
    max_features = 'sqrt'
    bootstrap = True
    min_samples_leaf = 2
    repeat = True
    vec_dim = 3
    times = 10
    mini_batch = True
    batch_size = 0.001
    lr = 0.1
    decay = 0

    run(
        n_samples,
        n_estimators, max_features, bootstrap,
        min_samples_leaf, repeat, vec_dim,
        times, mini_batch, batch_size, lr, decay)
