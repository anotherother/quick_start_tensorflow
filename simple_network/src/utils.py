from sklearn.datasets import fetch_20newsgroups
import numpy as np
from collections import Counter

def read_the_texts(print_shape=False):
    """
    Функция для чтения тектовых данных

    :param print_shape: 
    :return: 
    """
    categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    if print_shape == True:
        print('total texts in train:', len(newsgroups_train.data))
        print('total texts in test:', len(newsgroups_test.data))

    return newsgroups_train, newsgroups_test


def get_batch(df, i, batch_size, vocab):
    """
    Бьем текст на батчи
    :param df: 
    :param i: 
    :param batch_size: 
    :return: 
    """
    batches = []
    results = []

    total_words = len(vocab)
    word2index = get_word_2_index(vocab)

    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((3), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)

    return np.array(batches), np.array(results)



def create_vocab(newsgroups_train, newsgroups_test):

    vocab = Counter()

    for text in newsgroups_train.data:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    for text in newsgroups_test.data:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    return vocab


def get_word_2_index(vocab):
    """
    Функция получения индекса по слову
    :param vocab: 
    :return: 
    """
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index
