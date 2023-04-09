import argparse
import time
from collections import Counter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.corpus import brown, treebank, conll2000

nltk.download('brown')
nltk.download('treebank')
nltk.download('conll2000')

POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
            'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
            'SYM', 'VERB', 'X']
WORDS = None


def sep_sentences(words, sep):
    s = []
    curr = []
    for w in words:
        if w[0] != sep:
            curr.append(w)
        else:
            curr.append(w)
            s.append(curr)
            curr = []
    return s


def gen_prob_tables(train_lst):
    """
    Generate initial, transition and observation probability tables
    :param train_lst:
    :return:
    """
    global POS_TAGS, WORDS

    tokens = [(tup[0].lower(), tup[1]) for tup in train_lst]
    sentences = sep_sentences(tokens, '.')
    pos_seq = [[tup[1] for tup in s] for s in sentences]
    word_seq = [tup[0].lower() for tup in train_lst]

    vec = DictVectorizer(sparse=False)

    WORDS = list(set(word_seq))
    POS_TAGS = list(set([tup[1] for tup in train_lst]))
    POS_TAGS.sort()
    WORDS.sort()

    # Initial probabilities
    init_freq = dict(Counter([s[0] for s in pos_seq]))
    N = len(sentences)
    for p in POS_TAGS:
        if p not in init_freq:
            init_freq[p] = 0
    for k in init_freq:
        init_freq[k] /= N

    pos_initial = vec.fit_transform(init_freq).transpose()  # P x 1

    # Transition probabilities
    bigrams = dict(Counter([(s[i], s[i + 1]) for s in pos_seq for i in range(len(s) - 1)]))
    trans_freq = []
    for p in POS_TAGS:
        p_grams = {pos[1]: bigrams[pos] for pos in bigrams if pos[0] == p}
        for tag in POS_TAGS:
            if tag not in p_grams:
                p_grams[tag] = 0
        total = max(sum(p_grams.values()), 1)
        for k in p_grams:
            p_grams[k] /= total
        trans_freq.append(p_grams)

    pos_trans = vec.fit_transform(trans_freq)  # P x P

    # Observed probabilities
    word_freq = dict(Counter(word_seq))  # P(W)
    observed = {}
    counter = Counter(tokens)
    for pos in POS_TAGS:
        if pos not in observed:
            observed[pos] = {}
        for w in WORDS:
            observed[pos][w] = 0

    for (word, pos), val in counter.items():
        observed[pos][word] = val / word_freq[word]

    obs = []
    for pos in POS_TAGS:
        obs.append(observed[pos])
    word_obs = vec.fit_transform(obs)

    return pos_initial, pos_trans, word_obs


def pos_tagger(word_seq, pos_tags, initial, trans, obs):
    prob = np.zeros((len(word_seq), len(pos_tags)))
    prev = np.zeros((len(word_seq), len(pos_tags)))

    # time step 0
    prob[0] = initial.reshape(1, -1) * obs[:, word_seq[0]]
    prev[0] = None

    # time steps 1 to len(pos_tags) - 1
    for t in range(1, len(word_seq)):
        prev_vals = (prob[t - 1, :] * trans.T).T * obs[:, word_seq[t]]
        max_pos = np.argmax(prev_vals, axis=0)
        max_tile = np.tile(max_pos, (len(pos_tags), 1))
        X = np.take_along_axis(prev_vals, max_tile, 0)
        prob[t] = X[0]
        prev[t] = max_pos

    return prob, prev


def convert_pred(prob, prev):
    global POS_TAGS
    preds = []
    end = np.argmax(prob[len(prob) - 1])
    preds.append(POS_TAGS[end])
    for i in range(len(prob) - 1, 1, -1):
        ind = int(prev[i, end])
        preds.insert(0, POS_TAGS[ind])
        end = ind

    preds.insert(0, POS_TAGS[np.argmax(prob[0])])
    return preds


def write_to_file(filename, preds):
    f = open(filename, 'w')

    for pred in preds:
        for i in range(len(pred)):
            f.write(pred[i] + ' ')
        f.write('\n')


def encode_input(test):
    global WORDS
    res = []
    for s in test:
        words = []
        for w in s:
            ind = WORDS.index(w.lower())
            words.append(ind)

        res.append(words)
    return res


if __name__ == '__main__':

    word_pairs = brown.tagged_words(tagset='universal') + \
                 treebank.tagged_words(tagset='universal') + \
                 conll2000.tagged_words(tagset='universal')

    garden_test = []
    with open('data/data_garden.txt') as f:
        lines = f.readlines()

        for line in lines:
            tokens = line.split()
            tokens = [t.lower() for t in tokens]
            garden_test.append(tokens)

    normal_test = []
    with open('data/data_normal.txt') as f:
        lines = f.readlines()

        for line in lines:
            tokens = line.split()
            tokens = [t.lower() for t in tokens]
            normal_test.append(tokens)

    garden_labels = []
    with open('data/labels_garden.txt') as f:
        lines_out = f.readlines()
        for line in lines_out:
            tokens = line.split()
            garden_labels.append(tokens)

    normal_labels = []
    with open('data/labels_normal.txt') as f:
        lines_out = f.readlines()
        for line in lines_out:
            tokens = line.split()
            normal_labels.append(tokens)

    print("Generating probability tables...")

    I, T, M = gen_prob_tables(word_pairs)
    garden_sent = encode_input(garden_test)
    normal_sent = encode_input(normal_test)

    print("Starting the tagging process...")
    start_time = time.time()
    garden_preds = []
    for s in garden_sent:
        prob, prev = pos_tagger(s, POS_TAGS, I, T, M)
        pred = convert_pred(prob, prev)
        garden_preds.append(pred)

    normal_preds = []
    for s in normal_sent:
        prob, prev = pos_tagger(s, POS_TAGS, I, T, M)
        pred = convert_pred(prob, prev)
        normal_preds.append(pred)

    write_to_file('hmm/pred_garden.txt', garden_preds)
    write_to_file('hmm/pred_normal.txt', normal_preds)

    print(f"Time taken: {time.time() - start_time}")

    # garden_acc = [1 for i in range(len(garden_labels)) if garden_preds[i] == garden_labels[i]]
    # normal_acc = [1 for i in range(len(normal_labels)) if normal_preds[i] == normal_labels[i]]
    #
    # print("Tagging completed!")
    # print(f"Garden accuracy: {len(garden_acc) / len(garden_labels)}")
    # print(f"Normal accuracy: {len(normal_acc) / len(normal_labels)}")
