import csv
import os
import pickle
import gensim
from gensim.models import Doc2Vec
import random
import numpy as np
import re
from collections import defaultdict


def load_anew(filepath=None):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        words, arousal, valence = [], [], []
        for line in reader:
            words.append(line[0])
            valence.append(float(line[1]))
            arousal.append(float(line[2]))
    return words, valence, arousal


def load_extend_anew(D=False):
    print('Loading extend_anew lexicon')
    with open('./resource/extend_ANEW.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        words, arousal, valence, dominance = [], [], [], []
        for line in reader:
            if reader.line_num == 1:
                continue
            words.append(line[1])
            arousal.append(float(line[5]))
            valence.append(float(line[2]))
            if D == True:
                dominance.append(float(line[8]))
    print('Loading extend_anew lexicon complete')
    if D == True:
        return words, valence, arousal, dominance
    else:
        return words, valence, arousal


def load_Bing_Liu(polarity=None):
    if polarity == 'positive':
        filename = './resources/Lexicon/Bing_Liu/positive-words.txt'
    elif polarity == 'negative':
        filename = './resources/Lexicon/Bing_Liu/negative-words.txt'
    else:
        raise Exception('Wrong Argument.')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        words = []
        for line in reader:
            words.append(line[0])
            # print(line[0])
    return words


def load_csv(filename):
    out = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONE)
        for line in reader:
            out.append(line)
    return out


def load_sentiment140(filename):
    print('start loading data...')
    # ��ʽ��"4","1467822272","Mon Apr 06 22:22:45 PDT 2009","NO_QUERY","ersle","I LOVE @Health4UandPets u guys r the best!! "
    with open(filename, 'rt', encoding='ISO-8859-1') as f:
        inpTweets = csv.reader(f, delimiter=',', quotechar='"')
        X = []  # sentiment
        Y = []  # tweets
        for row in inpTweets:
            sentiment = (1 if row[0] == '4' else 0)
            tweet = row[5]
            X.append(sentiment)
            Y.append(tweet)
        # end loop
        return Y, X


def load_vader(filename):
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        texts, ratings = [], []
        for line in reader:
            texts.append(line[2])
            ratings.append(float(line[1]))
    return texts, ratings


def load_embeddings(arg=None, filename='None'):
    if arg == 'zh_tw':  # dim = 400
        model = gensim.models.Word2Vec.load_word2vec_format(None, binary=False)
    elif arg == 'google_news':
        model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)  # C binary format
    elif arg == 'CVAT':  # dim = 50
        model = gensim.models.Word2Vec.load(None)
    elif arg == 'twitter':  # dim = 50
        model = Doc2Vec.load('./data/acc/docvecs_twitter.d2v')
    else:
        raise Exception('Wrong Argument.')
    print('Load Model Complete.')
    return model


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out


def load_sst(path=None, level=None):
    def cleanStr(string):
        string = re.sub(r'^A-Za-z0-9(),!?\'\`', ' ', string)
        string = re.sub(r'\s{2,}', ' ', string)
        string = string.replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã±', 'ñ').replace('Â', '').replace('Ã¯', 'ï')
        string = string.replace('Ã¼', 'ü').replace('Ã¢', 'â').replace('Ã¨', 'è').replace('Ã¶', 'ö').replace('Ã¦', 'æ')
        string = string.replace('Ã³', 'ó').replace('Ã»', 'û').replace('Ã´', 'ô').replace('Ã£', 'ã').replace('Ã§', 'ç')
        string = string.replace('Ã ', 'à ').replace('Ã', 'í').replace('í­', 'í')
        return string

    # sentiment label
    sentiment_file = open(path + 'sentiment_labels.txt', 'r')
    sentiment_label = {}
    n = 0
    for line in sentiment_file:
        lines = line.strip().split('|')
        if n > 0:
            sentiment_label[int(lines[0])] = float(lines[1])
        n += 1
    sentiment_file.close()

    # phrase dict
    dict_file = open(path + 'dictionary.txt', 'r')
    phrase_dict = {}
    for line in dict_file:
        # line = line.decode('utf-8')
        lines = line.strip().split('|')
        phrase_dict[lines[0]] = int(lines[1])
    dict_file.close()

    # sentence dict
    sentence_file = open(path + 'datasetSentences.txt', 'r')
    sentence_dict = {}
    n = 0
    for line in sentence_file:
        # line = line.decode('utf-8')
        line = line.replace('-LRB-', '(')
        line = line.replace('-RRB-', ')')
        lines = line.strip().split('\t')
        if n > 0:
            sentence_dict[int(lines[0])] = lines[1]
        n += 1
    sentence_file.close()

    # datasplit
    datasplit_file = open(path + 'datasetSplit.txt', 'r')
    split_dict = {}
    n = 0
    for line in datasplit_file:
        lines = line.strip().split(',')
        if n > 0:
            split_dict[int(lines[0])] = int(lines[1])
        n += 1
    datasplit_file.close()
    size = len(sentence_dict)

    senti = sentiment_label[phrase_dict[cleanStr(sentence_dict[0 + 1])]]
    print(senti)
    # exit()

    for i in range(11854):
        try:
            print(sentiment_label[phrase_dict[cleanStr(sentence_dict[i + 1])]])
        except:
            print('*----' * 100)
            print(i, sentence_dict[i + 1], cleanStr(sentence_dict[i + 1]))
    exit()

    vec_file = open('vec_stanford.pkl', 'r')
    vecs = cPickle.load(vec_file)
    vec_file.close()

    x_train, y_train = [], []
    x_test, y_test = [], []
    x_valid, y_valid = [], []

    n0 = 0
    n1 = 1
    for i in range(len(vecs)):
        # print sentence_dict[i+1].encode('utf-8')

        senti = sentiment_label[phrase_dict[sentence_dict[i + 1]]]

        # print vecs[i]
        print(senti, sentence_dict[i + 1])
        if (senti > 0.4) and (senti <= 0.6):
            continue
        if senti > 0.6:
            senti = 1
            n1 += 1
        if senti <= 0.4:
            senti = 0
            n0 += 1
        if split_dict[i + 1] == 1:
            x_train.append(vecs[i])
            y_train.append(senti)
        elif split_dict[i + 1] == 2:
            x_test.append(vecs[i])
            y_test.append(senti)
        else:
            x_valid.append(vecs[i])
            y_valid.append(senti)

    print(len(x_train), len(x_valid), len(x_test))
    t = zip(x_train, y_train)
    random.shuffle(t)
    x_train, y_train = zip(*t)

    sentiment_trainingdata = open('sentiment_trainingdata.pkl', 'w')
    cPickle.dump(((x_train, y_train), (x_valid, y_valid), (x_test, y_test)), sentiment_trainingdata)
    sentiment_trainingdata.close()

    print(y_train)


def load_sst_2(path=None):
    def cleanStr(string):
        string = re.sub(r'^A-Za-z0-9(),!?\'\`', ' ', string)
        string = re.sub(r'\s{2,}', ' ', string)
        string = string.replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã±', 'ñ').replace('Â', '').replace('Ã¯', 'ï')
        string = string.replace('Ã¼', 'ü').replace('Ã¢', 'â').replace('Ã¨', 'è').replace('Ã¶', 'ö').replace('Ã¦', 'æ')
        string = string.replace('Ã³', 'ó').replace('Ã»', 'û').replace('Ã´', 'ô').replace('Ã£', 'ã').replace('Ã§', 'ç')
        string = string.replace('Ã  ', 'à ').replace('Ã', 'í').replace('í­', 'í')
        return string

    def loadSentences(fileName):
        Index2Sentence = {}
        Sentence2Index = {}
        with open(fileName, 'r') as fopen:
            i = 0
            for line in fopen:
                if i > 0:
                    parts = line.split('\t')
                    index = int(parts[0])
                    sentence = parts[1].replace('-LRB-', '(').replace('-RRB-', ')').replace('\n', '')
                    sentence = cleanStr(sentence)
                    Index2Sentence[index] = sentence
                    Sentence2Index[sentence] = index
                i += 1
        return Index2Sentence, Sentence2Index

    def lookupDict(dictFileName, Sentence2Index):
        Sentence2SentimentIndex = {}
        with open(dictFileName, 'r') as fopen:
            for line in fopen:
                parts = line.split('|')
                sentiment = parts[0].replace('-LRB-', '(').replace('-RRB-', ')').replace('\n', '')
                sentiment = cleanStr(sentiment)
                index = int(parts[1])
                if sentiment in Sentence2Index:
                    Sentence2SentimentIndex[sentiment] = index
        # assert len(Sentence2SentimentIndex) == len(Sentence2Index)
        for sentence in Sentence2Index:
            if not sentence in Sentence2SentimentIndex:
                print(sentence)
        return Sentence2SentimentIndex

    def loadLabels(sentimentLabelFile):
        SentimentIndex2Label = {}
        with open(sentimentLabelFile, 'r') as fopen:
            i = 0
            for line in fopen:
                if i > 0:
                    parts = line.split('|')
                    index = int(parts[0])
                    value = min(int(float(parts[1]) // 0.2), 4)
                    SentimentIndex2Label[index] = value
                i += 1
        return SentimentIndex2Label

    def loadSetLabel(setLabelFile):
        '''
        SetLabel: 1-train 2-test 3-dev
        '''
        Index2SetLabel = {}
        with open(setLabelFile, 'r') as fopen:
            i = 0
            for line in fopen:
                if i > 0:
                    parts = line.split(',')
                    index = int(parts[0])
                    setLabel = int(parts[1])
                    Index2SetLabel[index] = setLabel
                i += 1
        return Index2SetLabel

    def loadData(Sentence2Index, Index2Sentence, Sentence2SentimentIndex, SentimentIndex2Label, Index2SetLabel):
        vocab = defaultdict(float)
        sentences = []
        for sentence in Sentence2Index:
            index = Sentence2Index[sentence]
            sentimentIndex = Sentence2SentimentIndex[sentence]
            label = SentimentIndex2Label[sentimentIndex]
            setLabel = Index2SetLabel[index]
            clean = cleanStr(sentence)
            clean = clean.lower()
            words = set(clean.split())
            for word in words:
                vocab[word] += 1
            sentences.append({'label': label, 'text': clean.split(), 'setLabel': setLabel, 'len': len(clean.split())})
        return sentences, vocab

    fileName = path + 'datasetSentences.txt'
    dictFileName = path + 'dictionary.txt'
    sentimentLabelFile = path + 'sentiment_labels.txt'
    setLabelFile = path + 'datasetSplit.txt'
    Index2Sentence, Sentence2Index = loadSentences(fileName)
    Sentence2SentimentIndex = lookupDict(dictFileName, Sentence2Index)
    SentimentIndex2Label = loadLabels(sentimentLabelFile)
    Index2SetLabel = loadSetLabel(setLabelFile)
    sentences, vocab = loadData(Sentence2Index, Index2Sentence, Sentence2SentimentIndex, SentimentIndex2Label,
                                Index2SetLabel)
    # pickle.dump(
    #     [sentences, vocab, {'classes': 5, 'all': [1, 2, 3], 'train': [1], 'test': [2], 'dev': [3], 'cross': False}],
    #     open('data', 'wb'))
    print('data processed')
