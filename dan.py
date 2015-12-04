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



def dan_original():
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


if __name__ == "__main__":
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

    model = dan_original()

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")

    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3, validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)