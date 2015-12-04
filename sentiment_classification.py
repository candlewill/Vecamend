from load_data import load_sst

if __name__ == '__main__':
    (x_train, y_train_valence, y_train_labels,
    x_test, y_test_valence, y_test_labels,
    x_valid, y_valid_valence, y_valid_labels,
    x_train_polarity, y_train_polarity,
    x_test_polarity, y_test_polarity,
    x_valid_polarity, y_valid_polarity) = load_sst(path='./resources/stanfordSentimentTreebank/')

    for i in x_train:
        print(i)