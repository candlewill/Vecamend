from load_data import load_anew
from load_data import load_embeddings

def build_ori_anew_vectors(words):
    model = load_embeddings('google_news', '/home/hs/Data/Word_Embeddings/google_news.bin')
    full_words = model.vocab.keys()
    same_words = set(words).intersection(full_words)
    print(set(words)-same_words)
    print(len(same_words))

if __name__=='__main__':
    words, valence, arousal = load_anew('./resources/Lexicon/ANEW.txt')
    for i in words:
        print(i)
    print(len(words))
    build_ori_anew_vectors(words)