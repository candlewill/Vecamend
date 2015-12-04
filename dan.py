from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.constraints import unitnorm
from sentiment_classification import build_keras_input

'''
    Train a deep averaging network (DAN) using keras.
    The model is described in "Deep Unordered Composition Rivals Syntactic Methods
    for Text Classification" by Iyyer, Manjunatha, Boyd-Graber, and Daume (ACL 2015).
    Notes:
    - Deep averaging networks can be thought of as a drop-in for recurrent neural networks
    - In the paper, dropout was applied at the word level; that's not done here. I just added dropout
    in the fully connected layers. This could be fixed later.
    - The sequence of characters doesn't matter, so it's pretty surprising that something like
    this does well.
    - Parameters are not optimized in any way -- I just used a fixed number for the embedding
    and hidden dimension
    This model achieves 0.8340 test accuracy after 3 epochs for IMDB sentiment classification.
    Input Data format:
        X: a list of number indicating the word index, e.g. [12 54 63 74 12 10]
        Y: a list of 0 or 1 indicating the labels of sentence, e.g. [0 1 0 1 0 1 0 0 1]
'''



def dan_original(max_features):
    '''
    DAN model
    :param max_features: the number of words
    :return: keras model
    '''
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 300))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=1, activation = 'sigmoid'))
    return model

def dan_pre_trained(max_features, weights=None):
    '''
    DAN model with pre-trained embeddings
    :param max_features: the number of words
    :return: keras model
    '''
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 300, weights=[weights]))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=1, activation = 'sigmoid'))
    return model

def dan_simplified(max_features, weights=None):
    '''
    DAN model with pre-trained embeddings, just use one non-linear layer
    :param max_features: the number of words
    :return: keras model
    '''
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 300, weights=[weights]))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=1, activation = 'sigmoid'))
    return model

def dan_dropout(max_features, weights=None):
    '''
    DAN model with pre-trained embeddings, the position of dropout is changed
    :param max_features: the number of words
    :return: keras model
    '''
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 300, weights=[weights]))
    model.add(Dropout(.5))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=1, activation = 'sigmoid'))
    return model

def dan_dropout_p(max_features, weights=None):
    '''
    DAN model with pre-trained embeddings, the position of dropout is changed and the dropuout rate is 0.3
    :param max_features: the number of words
    :return: keras model
    '''
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 300, weights=[weights]))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(input_dim=300, output_dim=1, activation = 'sigmoid'))
    return model

def test_dan_original():
    max_features = 20000
    maxlen = 100  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')


    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    model = dan_original(max_features)

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")

    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3, validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)

def SA_sst():
    ((x_train_idx_data, y_train_valence, y_train_labels,
     x_test_idx_data, y_test_valence, y_test_labels,
     x_valid_idx_data, y_valid_valence, y_valid_labels,
     x_train_polarity_idx_data, y_train_polarity,
     x_test_polarity_idx_data, y_test_polarity,
     x_valid_polarity_idx_data, y_valid_polarity), W) = build_keras_input()


    maxlen = 200  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    (X_train, y_train), (X_test, y_test), (X_valid, y_valide) = (x_train_polarity_idx_data, y_train_polarity), (x_test_polarity_idx_data, y_test_polarity), (x_valid_polarity_idx_data, y_valid_polarity)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    # m= 0
    # for i in X_train:
    #     if len(i) >0:
    #         for j in i:
    #             if j > m:
    #                 m=j
    # print(m)
    max_features = W.shape[0] # shape of W: (13631, 300)

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    model = dan_dropout_p(max_features, W)

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")

    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50, validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)



if __name__ == "__main__":
    SA_sst()