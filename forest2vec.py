import numpy as np
import tensorflow as tf
from tensorflow import keras
import multiprocessing as mp
import time
import datetime
import itertools
from sklearn.ensemble import RandomForestClassifier
import sys
import json


class SkipGram(keras.Model):
    def __init__(self, v_dim, emb_dim, lr=0.01, decay=0.0001):
        super().__init__()
        self.v_dim = v_dim
        self.emb_dim = emb_dim
        self.embeddings = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
        )
        self.nce_w = self.add_weight(
            name="nce_w", shape=[v_dim, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1))
        self.nce_b = self.add_weight(
            name="nce_b", shape=(v_dim,),
            initializer=keras.initializers.Constant(0.1))

        self.opt = keras.optimizers.Adam(lr=lr, decay=decay)

    def call(self, x, training=None, mask=None):
        o = self.embeddings(x)
        return o

    def get_loss(self, x, y, training=None):
        embedded = self.call(x, training)
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_w, biases=self.nce_b, labels=tf.expand_dims(y, axis=1),
                inputs=embedded, num_sampled=5, num_classes=self.v_dim))

    def step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.get_loss(x, y, True)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


class DRLForest():
    def __init__(self, timer=False, print_loss=False):
        self.timer = timer
        self.print_loss = print_loss
        self.loss = []

    def _check_bootstrap(self, X_train, rfc):
        total = [0] * X_train.shape[0]
        for tree in rfc.estimators_:
            total += tree.sample_counts
        not_sampled = False
        for t in total:
            if t == 0:
                not_sampled = True
        if not_sampled:
            raise ValueError("some samples are not sampled, increase the number of trees")

    def _prune(self, rfc, X_train, y_train, min_samples_leaf=2):
        params = rfc.get_params()
        params['min_samples_leaf'] = min_samples_leaf
        rfc_pruned = RandomForestClassifier(**params)
        rfc_pruned.fit(X_train, y_train)
        return rfc_pruned

    def _leaf2groups(self, leaf_index, sampled_index=None, sample_counts=None, repeat=True):
        leaf_index_unique = np.unique(leaf_index)
        groups_dic = {}
        for leaf in leaf_index_unique:
            sample_leaf = np.where(leaf_index == leaf)[0]
            sample_leaf = sampled_index[sample_leaf]
            if repeat:
                a = sample_counts[sample_leaf]
                b = sample_leaf.repeat(a)
                sample_leaf = b
                sample_leaf = sample_leaf.repeat(sample_counts[sample_leaf])
            sample_leaf = sample_leaf.tolist()  # new int
            groups_dic[leaf] = sample_leaf
        return groups_dic

    def get_groups(self, X, rfc, repeat=True, save_groups=False):
        self._check_bootstrap(X, rfc)
        groups_rfc = []
        for dtc in rfc.estimators_:
            sampled_index = np.where(dtc.sample_counts != 0)[0]
            X_sampled = X[sampled_index]
            leaf_index = dtc.apply(X_sampled)
            groups_dic = self._leaf2groups(leaf_index, sampled_index, dtc.sample_counts, repeat=repeat)
            if save_groups:
                dtc.groups = groups_dic
            groups_rfc.extend(list(groups_dic.values()))
        return groups_rfc

    def _pair_group(self, group):
        pair = np.array(list(itertools.permutations(group, 2)), dtype=np.uint16)
        same = []
        for i in range(len(pair)):
            if pair[i][0] == pair[i][1]:
                same.append(i)
        pair = np.delete(pair, same, axis=0)
        return pair

    def prepare_data(self, groups):
        start = time.time()
        pool = mp.Pool(processes=mp.cpu_count()-2, maxtasksperchild=1)
        pair = pool.map(self._pair_group, groups)
        pool.close()
        middle = time.time()
        if self.timer:
            print("pair_group:", str(datetime.timedelta(seconds=int(middle - start))))
        pairs = np.concatenate(pair)
        print(sys.getsizeof(pairs) / 1024 / 1024, 'MB')
        u, v = pairs[:, 0], pairs[:, 1]
        end = time.time()
        if self.timer:
            print("merge_pair:", str(datetime.timedelta(seconds=int(end - middle))))
        return u, v

    def train(self, model, u, v, times=5, mini_batch=True, batch_size=10000):
        start = time.time()
        if batch_size < 1:
            batch_size = int(len(u) * batch_size)
        self.batch_size = batch_size
        if times < 1000:
            times = int(len(u) * times / batch_size)
        self.times = times
        self.uv_len = len(u)
        for t in range(times):
            if mini_batch:
                random_id = np.random.randint(0, len(u), batch_size)
                u_batch, v_batch = u[random_id], v[random_id]
                loss = model.step(u_batch, v_batch)
            else:
                loss = model.step(u, v)
            if self.print_loss:
                print("step: {} / {} | loss: {}".format(t+1, times, loss))
            self.loss.append(loss)
        end = time.time()
        if self.timer:
            print("train:", str(datetime.timedelta(seconds=int(end - start))))

    def fit_transform(self, X_train, y_train, rfc, min_samples_leaf=2, vec_dim=2,
                      times=5, mini_batch=True, batch_size=10000, lr=0.01, decay=0.0001,
                      repeat=True, multiprocessing=False, order=None, num=None):
        self.get_groups(X_train, rfc, repeat=repeat, save_groups=True)
        if type(min_samples_leaf) == int:
            rfc_pruned = self._prune(rfc, X_train, y_train, min_samples_leaf=min_samples_leaf)
            groups_rfc = self.get_groups(X_train, rfc_pruned, repeat=repeat, save_groups=False)
        else:
            groups_rfc = []
            for min_samples_leaf_ in min_samples_leaf:
                rfc_pruned = self._prune(rfc, X_train, y_train, min_samples_leaf=min_samples_leaf_)
                groups_rfc_ = self.get_groups(X_train, rfc_pruned, repeat=repeat, save_groups=False)
                groups_rfc.extend(groups_rfc_)
        u, v = self.prepare_data(groups_rfc)
        model = SkipGram(X_train.shape[0], vec_dim, lr=lr, decay=decay)
        self.train(model, u, v, times=times, mini_batch=mini_batch, batch_size=batch_size)
        self.model = model
        embedding_vec = model.embeddings.get_weights()[0]
        if multiprocessing:
            print(order, '/', num)
        return embedding_vec

    def _get_mean_dtc(self, X_train_embedding, rfc):
        for dtc in rfc.estimators_:
            mean_dic = {}
            for key, value in zip(dtc.groups.keys(),
                                  dtc.groups.values()):
                sum_leaf = np.zeros(shape=(X_train_embedding.shape[1]))
                for v in value:
                    sum_leaf = sum_leaf + X_train_embedding[v]
                mean_leaf = sum_leaf / len(value)
                mean_dic[key] = mean_leaf
            dtc.mean_leaf = mean_dic

    def transform(self, X_test, X_train_embedding, rfc):
        self._get_mean_dtc(X_train_embedding, rfc)
        X_test_leaf = rfc.apply(X_test)
        X_test_embedding = np.zeros(shape=(X_test.shape[0], X_train_embedding.shape[1]))
        for i in range(len(X_test_leaf)):
            sum_dtc = np.zeros(shape=(X_train_embedding[0].shape))
            for leaf, dtc in zip(X_test_leaf[i], rfc.estimators_):
                sum_dtc = sum_dtc + dtc.mean_leaf[leaf]
            mean_dtc = sum_dtc / len(rfc.estimators_)
            X_test_embedding[i] = mean_dtc
        return X_test_embedding