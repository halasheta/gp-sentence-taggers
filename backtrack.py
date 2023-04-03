import nltk
from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.corpus import brown, treebank, conll2000
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, FreqDist

nltk.download('brown')
nltk.download('treebank')
nltk.download('conll2000')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

txt = brown.tagged_words(tagset='universal') + \
      treebank.tagged_words(tagset='universal') + \
      conll2000.tagged_words(tagset='universal')

tokens = [(tup[0].lower(), tup[1]) for tup in txt]
pos_lst = [tup[1] for tup in tokens]

# Make a word-frequency dict
counter = Counter(tokens)
word_freq = {}
for (k1, k2), value in counter.items():
    if k1 not in word_freq:
        word_freq[k1] = {}
    word_freq[k1][k2] = value


def pos_predict(s, assignment, threshold):
    if len(assignment) == len(s):
        # Base case: assignment is returned if it is complete
        return assignment

    var = s[len(assignment)].lower()
    var_fdist = FreqDist(word_freq[var])
    var_freq = var_fdist.most_common()  # Ordered by descending frequency

    for pos in var_freq:
        # for each tag, append it to assignment
        assignment.append(pos[0])

        valid = True
        if len(assignment) > 1:
            # Check if curr sequence of POS predictions match the frequencies from the text
            window = len(assignment)
            ngrams = list(nltk.ngrams(pos_lst, window))
            cfdist = ConditionalFreqDist((tuple(x[-1 * window:-1]), x[-1]) for x in ngrams)
            cpdist = ConditionalProbDist(cfdist, nltk.MLEProbDist)

            fdist = cpdist[tuple(assignment[-1 * window:-1])].freqdist()
            cond_prob = fdist.freq(assignment[-1])

            surpr = (-1 * np.log2(max(0.00001, cond_prob))) / (window - 1)
            if round(surpr, 2) > threshold:
                valid = False

        # predict the rest of the sentence based on curr tag
        if valid:
            res = pos_predict(s, assignment, threshold)
            if len(res) > 0:
                return res

        # if assignment is fault, pop and try others
        assignment.pop()

    return []


def backtrack_model(sentences, threshold):
    pred = []
    for i in tqdm(range(len(sentences))):
        p = pos_predict(sentences[i], [], threshold)
        pred.append(p)
    return pred


garden_data = []
with open('data/data_garden.txt') as f:
    lines = f.readlines()

    for line in lines:
        tokens = line.split()
        tokens = [t.lower() for t in tokens]
        garden_data.append(tokens)

garden_labels = []
with open('data/labels_garden.txt') as f:
    lines_out = f.readlines()
    for line in lines_out:
        tokens = line.split()
        garden_labels.append(tokens)

normal_data = []
with open('data/data_normal.txt') as f:
    lines = f.readlines()

    for line in lines:
        tokens = line.split()
        tokens = [t.lower() for t in tokens]
        normal_data.append(tokens)

normal_labels = []
with open('data/labels_normal.txt') as f:
    lines_out = f.readlines()
    for line in lines_out:
        tokens = line.split()
        normal_labels.append(tokens)

sup_range = np.arange(1.3, 2, 0.05)

print("Predicting garden path sentences...")

f_out = open('btrk/pred_garden.txt', 'w')
for num in tqdm(sup_range):
    pred = backtrack_model(garden_data, num)
    for s in pred:
        for pos in s:
            f_out.write(pos + " ")
        f_out.write('\n')
    f_out.write(',')
    f_out.write('\n')
f_out.close()

print("Done!")

print("Predicting normal sentences...")

f_out = open('btrk/pred_normal.txt', 'w')
for num in tqdm(sup_range):
    pred = backtrack_model(normal_data, num)
    for s in pred:
        for pos in s:
            f_out.write(pos + " ")
        f_out.write('\n')
    f_out.write(',')
    f_out.write('\n')
f_out.close()

print("Done!")