import forest2vec as fv
# import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# class T_train:
#     times = 0
#
# class T_test:
#     times = 0


def get_mnist_data():
    from sklearn import datasets
    digits = datasets.load_digits()
    data = digits.data
    label = digits.target
    # class_names = digits.target_names
    return data, label


def get_splited_mnist_data(sample_discarded_percent=0, test_size=0.5):  # 为了好展示，可以扔掉九成的数据
    data_all, label_all = get_mnist_data()
    from sklearn.model_selection import train_test_split
    data, data_, label, label_ = train_test_split(data_all, label_all, test_size=sample_discarded_percent,
                                                  random_state=0, stratify=label_all)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size,
                                                        random_state=1, stratify=label)
    return X_train, X_test, y_train, y_test

def get_imdb_data():
    from keras.datasets import imdb
    word_to_index = imdb.get_word_index()
    index_to_word = [None] * (max(word_to_index.values()) + 1)
    for w, i in word_to_index.items():
        index_to_word[i] = w
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                          num_words=10000,
                                                          skip_top=10,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)
    X_train = [
        ' '.join(
            index_to_word[i]
            for i in X_train[i]
            if i < len(index_to_word)
        ) for i in range(X_train.shape[0])
    ]

    X_test = [
        ' '.join(
            index_to_word[i]
            for i in X_test[i]
            if i < len(index_to_word)
        ) for i in range(X_test.shape[0])
    ]

    X_all = X_train + X_test  # Combine both to fit the TFIDF vectorization.
    lentrain = len(X_train)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_all = vectorizer.fit_transform(X_all)
    X_train = X_all[:lentrain]  # Separate back into training and test sets.
    X_test = X_all[lentrain:]
    return (X_train.toarray(), y_train), (X_test.toarray(), y_test)


def train_a_tree_and_get_leaf_index(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    # 叶结点里面不能只有一个样本，这样就不知道样本之间的关系了
    dtc = DecisionTreeClassifier(min_samples_leaf=2).fit(X_train, y_train)
    leaf_index = dtc.apply(X_train)
    return dtc, leaf_index


def get_sentences_by_many_trees(X_train, y_train, num_trees=10, merge=False):
    sentences = []
    windows = []
    for i in range(num_trees):
        dtc, leaf_index = train_a_tree_and_get_leaf_index(X_train, y_train)
        sentence, window = fv.leaf2sent(leaf_index)
        if merge:
            sentences.extend(sentence)
        else:
            sentences.append(sentence)
        windows.append(window)
    max_window = max(windows)
    if merge:
        return sentences, max_window
    else:
        multi_sentences = sentences
        return multi_sentences, windows


def get_parameters(rfc, multi_sentences, sample_num,
                   vec_dim=2, times=5000, sample_percnet=0.05, printer=False, multiprocessing=False):
    n_estimators =  rfc.get_params()['n_estimators']
    sample_nums = [sample_num] * n_estimators
    vec_dims = [vec_dim] * n_estimators
    timeses = [times] * n_estimators
    sample_percnets = [sample_percnet] * n_estimators
    printers = [printer] * n_estimators
    multiprocessinges = [multiprocessing] *n_estimators
    orders = list(range(n_estimators))
    nums = [n_estimators] * n_estimators
    parameters = list(zip(multi_sentences, sample_nums, vec_dims,
                          timeses, sample_percnets, printers, multiprocessinges,
                          orders, nums))
    return parameters


def get_parameters_new(rfc, multi_sentences, sample_num,
                   vec_dim=2, epochs=10, batch_size=1000, printer=False, times=10000,
                       sample_percent=0.05, multiprocessing=False):
    n_estimators =  rfc.get_params()['n_estimators']
    sample_nums = [sample_num] * n_estimators
    vec_dims = [vec_dim] * n_estimators
    epochses = [epochs] * n_estimators
    batch_sizes = [batch_size] * n_estimators
    printers = [printer] * n_estimators
    timeses = [times] * n_estimators
    sample_percents = [sample_percent] * n_estimators
    multiprocessings = [multiprocessing] * n_estimators
    orders = list(range(n_estimators))
    nums = [n_estimators] * n_estimators
    parameters = list(zip(multi_sentences, sample_nums, vec_dims, epochses,
                          batch_sizes, printers, timeses, sample_percents, multiprocessings, orders, nums))
    return parameters


# def show_w2v_word_embedding(model, data: fv.Dataset, path):  # movan data
#     word_emb = model.embeddings.get_weights()[0]
#     for i in range(data.num_word):
#         c = "blue"
#         try:
#             int(data.i2v[i])
#         except ValueError:
#             c = "red"
#         plt.text(word_emb[i, 0], word_emb[i, 1], s=data.i2v[i], color=c, weight="bold")
#     plt.xlim(word_emb[:, 0].min() - .5, word_emb[:, 0].max() + .5)
#     plt.ylim(word_emb[:, 1].min() - .5, word_emb[:, 1].max() + .5)
#     plt.xticks(())
#     plt.yticks(())
#     plt.xlabel("embedding dim1")
#     plt.ylabel("embedding dim2")
#     plt.savefig(path, dpi=300, format="png")
#     plt.show()


def plot_2D(data, label, path):  # mnist
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    ax.set_xlim(max(data[:, 0]), min(data[:, 0]))
    ax.set_ylim(max(data[:, 1]), min(data[:, 1]))
    plt.savefig(path, dpi=300, format="pdf")
    # plt.show()


def plot_3D(data, label, path):  # mnist
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(label[i]),
                color=plt.cm.Set1(label[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    ax.set_xlim(max(data[:,0]), min(data[:,0]))
    ax.set_ylim(max(data[:,1]), min(data[:,1]))
    ax.set_zlim(max(data[:,2]), min(data[:,2]))
    plt.savefig(path, dpi=300, format="pdf")
#     plt.show()


def plot_tree(dtc):
    from sklearn import tree
    plt.figure(dpi=300)
    tree.plot_tree(dtc, filled=True, node_ids=True)
    plt.show()
    # 同一路径上节点按数字编号大小排列


def check_dir(path):  # circle
    import os
    import os.path as osp
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path))


def plot2d(X, color, save_path=None):  # circle
    plt.scatter(X[:, 0], X[:, 1], c=color, s=4, cmap=plt.cm.Spectral)
    if save_path is None:
        plt.show()
    else:
        print("[plot2d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path, dpi = 300)
        plt.clf()
        plt.close('all')


def plot3d(X, color, save_path=None):  # circle
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    if save_path is None:
        plt.show()
    else:
        print("[plot3d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path, dpi = 300)
        plt.clf()
        plt.close('all')


# def set_seed(seed):  # circle
#     # from .log_utils import logger
#     if seed is None:
#         seed = np.random.randint(10000)
#     np.random.seed(seed)
#     # try:
#     #     import torch
#     #     torch.manual_seed(seed)
#     # except ImportError:
#     #     pass
#     # logger.info("[set_seed] seed={}".format(seed))
#     return seed


def show_confusion_matrix(X, y, rfc, path):
    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(rfc, X, y, display_labels=rfc.classes_, cmap=plt.cm.Blues, normalize=None)
    plt.savefig(path, dpi=300, format="pdf")


def get_score(X, X_vec, y, clf):
    from sklearn import metrics
    accuracy = clf.score(X, y)
    SC_old = metrics.silhouette_score(X, y, metric='euclidean')
    SC_new = metrics.silhouette_score(X_vec, y, metric='euclidean')
    CH_old = metrics.calinski_harabasz_score(X, y)
    CH_new = metrics.calinski_harabasz_score(X_vec, y)
    DB_old = metrics.davies_bouldin_score(X, y)
    DB_new = metrics.davies_bouldin_score(X_vec, y)
    return accuracy, SC_old, SC_new, CH_old, CH_new, DB_old, DB_new

def get_score_un(X, X_vec, y):
    from sklearn import metrics
    # accuracy = rfc.score(X, y)
    SC_old = metrics.silhouette_score(X, y, metric='euclidean')
    SC_new = metrics.silhouette_score(X_vec, y, metric='euclidean')
    CH_old = metrics.calinski_harabasz_score(X, y)
    CH_new = metrics.calinski_harabasz_score(X_vec, y)
    DB_old = metrics.davies_bouldin_score(X, y)
    DB_new = metrics.davies_bouldin_score(X_vec, y)
    return SC_old, SC_new, CH_old, CH_new, DB_old, DB_new


def plot_loss(loss, save_path=None):
    epochs = range(len(loss))
    plt.figure(figsize=(20, 4))
    plt.plot(epochs, loss, 'b', linewidth=0.2, label='Training loss')
    plt.title('Training loss')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        print("[plot_loss] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.clf()
        plt.close('all')