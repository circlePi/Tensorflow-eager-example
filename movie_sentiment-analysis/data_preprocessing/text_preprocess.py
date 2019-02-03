import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def text_clean():
    def review_to_wordlist(review):
        review_text = BeautifulSoup(review).get_text()
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        words = review_text.lower().split()
        return words

    train = pd.read_csv('./datasets/Imdb.tsv', sep='\t', header=0, quoting=3)
    labels = train['sentiment'].tolist()
    train['review'] = train['review'].apply(lambda s: review_to_wordlist((s)))
    reviews = train['review'].tolist()
    return reviews, labels


def word_to_id(word, word2id):
    return word2id[word] if word in word2id else 'unk'


def bulid_vocab(min_freq=1, stop_list=None, test_size=0.3, train_use_size=None):
    reviews, labels = text_clean()
    count = Counter()
    for review in reviews:
        count.update(review)
    if stop_list:
        count = {k: v for k, v in count.items() if k not in stop_list}
    # 词典
    vocab = [w for w, c in count.items() if c >= min_freq]
    vocab += ['unk']
    # 词典到编号的映射
    word2id = {k: v for k, v in zip(vocab, range(1, len(vocab)+1))}
    reviews = [[word_to_id(word, word2id) for word in review] for review in tqdm(reviews)]
    if train_use_size:
        reviews = reviews[:train_use_size]
    x_train, x_valid, y_train, y_valid = train_test_split(reviews,
                                                          labels,
                                                          random_state=2018,
                                                          test_size=test_size,
                                                          shuffle=True)
    return vocab, word2id, x_train, x_valid, y_train, y_valid


if __name__ == '__main__':
    reviews, labels = text_clean()
    bulid_vocab(reviews)






