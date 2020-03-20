
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
from funcy import lflatten
import re


def s2dict(lines, lot):

    """
    :param lines: list of sentences or words as strings containing at least two nodes to be mapped in dict
    :param lot: list of tags to be mapped in the dictionary as keys
    :return: dict with keys == tag and values == sentences /words
    """
    d = defaultdict(list)
    for line_, tag_ in zip(lines, lot):
        d[tag_] = line_

    return d



def make_causal_input(lod, map_, silent=True):
    #TODO replace hardcoded path by map_

    """
    :param lod: list of dictionaries
    :param map_: mapping of tags and values of interest, i.e. [('cause', 'C'), ('effect', 'E')]. The silent tags are by default taggerd as '_'
    :return: dict of list of tuples for each sentence
    """

    dd = defaultdict(list)
    dd_ = []
    rx = re.compile(r"(\b[-']\b)|[\W_]")
    rxlist = [r'("\\)', r'(\\")']
    rx = re.compile('|'.join(rxlist))
    for i in range(len(lod)):
        line_ = lod[i]['sentence']
        line = re.sub(rx, '', line_)
        #print(line)
        ante = lod[i]['cause']
        ante = re.sub(rx, '', ante)
        cons = lod[i]['effect']
        cons = re.sub(rx, '', cons)

        silent or print(line)
        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(word_tokenize(line)):
            index = line.find(w, index)

            if not index == -1:
                d[idx].append([w, index])
                silent or print(w, index)

                index += len(w)

        d_= defaultdict(list)
        for idx in d:

            d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])

            init_a = line.find(ante)
            init_c = line.find(cons)

            for el in word_tokenize(ante):
                start = line.find(el, init_a)
                # print('start A')
                # print(start)
                # print(int(d_[idx][0][1]))
                stop = line.find(el, init_a) + len(el)
                word = line[start:stop]
                #print(word)
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'C']), line.find(word, init_a)])
                    d_[idx] = und_[idx]
                init_a += len(word)


            for el in word_tokenize(cons):

                start = line.find(el, init_c)
                # print('start C')
                # print(start)
                # print(int(d_[idx][0][1]))
                stop = line.find(el, init_c) + len(el)
                word = line[start:stop]
                #print(word)
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'E']), line.find(word, init_c)])
                    d_[idx] = und_[idx]
                init_c += len(word)

        dd[i].append(d_)

    for dict_ in dd:
        dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])

    return dd_

def nltkPOS(loft):

    su_pos = []
    exceptions = []
    for i, j in enumerate(loft):

        try:
            text = str(j)
            pos_ = list(nltk.pos_tag(text.split()))
            su_pos.append(pos_)
        except:
            exceptions.append(i)

    tokenized_text = [word_tokenize(str(k)) for k in loft]
    overlist = [tokenized_text[l] for l in set(exceptions)]
    new_tok = [e for e in tokenized_text if e not in overlist]

    if not len(su_pos) == len(new_tok):
        print('concurrent size warning, will not be able to concat dictionaries')
    else:
        print('sizing is ok')

    return su_pos, exceptions



# ##  PREPARE MORE FEATURES

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]


# A function fo generating the list of labels for each document: TOKEN, POS, LABEL
def get_multi_labels(doc):
    return [label for (token, postag, label) in doc]





if __name__ == '__main__':

    import pandas as pd


    #df = pd.read_csv("FNP_2020_FinCausal/baseline/task2/data/fnp2020-fincausal2-task2.csv", delimiter=';', header=0)
    df = pd.read_csv("./data/fnp2020-fincausal2-task2.csv", delimiter=';', header=0)

    print(df.head())
    print(df.columns)

    lodict_ = []
    for rows in df.itertuples():
        list_ = [rows[2], rows[3], rows[4]]
        map1 = ['sentence', 'cause', 'effect']
        dict_ = s2dict(list_, map1)
        lodict_.append(dict_)

    print(lodict_[1])

    map_ = [('cause', 'C'), ('effect', 'E')]
    hometags = make_causal_input(lodict_, map_)
    postags, _ = nltkPOS([i['sentence'] for i in lodict_])



