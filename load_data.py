import csv
import os
import pickle
import gensim
from gensim.models import Doc2Vec
import random


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
    with open(filename,'r') as f:
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

def load_sst(path=None):
    # sentiment label
    sentiment_file = open(path+'sentiment_labels.txt', 'r')
    sentiment_label = {}
    n = 0
    for line in sentiment_file:
        lines = line.strip().split('|')
        if n > 0:
            sentiment_label[int(lines[0])] = float(lines[1])
        n += 1
    sentiment_file.close()

    # phrase dict
    dict_file = open(path+'dictionary.txt', 'r')
    phrase_dict = {}
    for line in dict_file:
        # line = line.decode('utf-8')
        lines =line.strip().split('|')
        phrase_dict[lines[0]] = int(lines[1])
    dict_file.close()

    #sentence dict
    sentence_file = open(path+'datasetSentences.txt', 'r')
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
    datasplit_file = open(path+'datasetSplit.txt', 'r')
    split_dict = {}
    n = 0
    for line in datasplit_file:
        lines = line.strip().split(',')
        if n > 0:
            split_dict[int(lines[0])] = int(lines[1])
        n += 1
    datasplit_file.close()

    senti = sentiment_label[phrase_dict[sentence_dict[102+1]]]
    print(senti)
    # exit()

    for i in range(11854):
        try:
            sentiment_label[phrase_dict[sentence_dict[i+1]]]
        except:
            print('*----'*100)
            print(i, sentence_dict[i+1])
    exit()


    vec_file = open('vec_stanford.pkl','r')
    vecs = cPickle.load(vec_file)
    vec_file.close()

    x_train, y_train = [], []
    x_test, y_test = [], []
    x_valid, y_valid = [], []

    n0 = 0
    n1 = 1
    for i in range(len(vecs)):
        #print sentence_dict[i+1].encode('utf-8')
        try:
            senti = sentiment_label[phrase_dict[sentence_dict[i+1]]]
        except:
            print(sentence_dict[i+1])
            continue
        #print vecs[i]
        print(senti, sentence_dict[i+1])
        if (senti > 0.4) and (senti <= 0.6):
            continue
        if senti > 0.6:
            senti = 1
            n1 += 1
        if senti <= 0.4:
            senti = 0
            n0 += 1
        if split_dict[i+1] == 1:
            x_train.append(vecs[i])
            y_train.append(senti)
        elif split_dict[i+1] == 2:
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
    cPickle.dump(((x_train,y_train), (x_valid, y_valid), (x_test, y_test)), sentiment_trainingdata)
    sentiment_trainingdata.close()

    print(y_train)