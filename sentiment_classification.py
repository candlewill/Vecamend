from load_data import load_sst
import numpy as np
from collections import defaultdict
from load_data import load_embeddings

def clean_str(sent):
    sent = sent.strip().replace('.', '').replace(',', '')
    sent = sent.replace(';', '').replace('<br />', ' ')
    sent = sent.replace(':', '').replace('"', '')
    sent = sent.replace('(', '').replace(')', '')
    sent = sent.replace('!', '').replace('*', '')
    sent = sent.replace(' - ', ' ').replace(' -- ', '')
    sent = sent.replace('?', '')
    sent = sent.lower()
    return ' '.join(sent.split())


def get_vocab(corpus):
    vocab = defaultdict(int)
    for sent in corpus:
        for word in clean_str(sent).split():
            vocab[word] += 1
    print('The total number of vocabulary is: %s. ' % len(vocab))
    return vocab

# word_vecs is the model of word2vec
def build_embedding_matrix(word_vecs, vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    union = (set(word_vecs.vocab.keys()) & set(vocab.keys()))
    vocab_size = len(union)
    print('The number of words occuring in corpus and word2vec simutaneously: %s.' %vocab_size)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k, dtype=np.float32)
    for i, word in enumerate(union, start=1):
        print(word, i)
        W[i] = word_vecs[word]
        word_idx_map[word] = i  # dict
    return W, word_idx_map

def sent2ind(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x


def make_idx_data(sentences, word_idx_map):
    """
    Transforms sentences (corpus, a list of sentence) into a 2-d matrix.
    """
    idx_data = []
    for sent in sentences:
        idx_sent = sent2ind(sent, word_idx_map)
        idx_data.append(idx_sent)
    idx_data = np.array(idx_data, dtype=np.int)
    return idx_data




if __name__ == '__main__':
    # load data from pickle
    (x_train, y_train_valence, y_train_labels,
    x_test, y_test_valence, y_test_labels,
    x_valid, y_valid_valence, y_valid_labels,
    x_train_polarity, y_train_polarity,
    x_test_polarity, y_test_polarity,
    x_valid_polarity, y_valid_polarity) = load_sst(path='./resources/stanfordSentimentTreebank/')

    vocab = get_vocab(x_train)
    word_vecs = load_embeddings()
