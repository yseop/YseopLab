
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
        caus = lod[i]['cause']
        caus = re.sub(rx, '', caus)
        effe = lod[i]['effect']
        effe = re.sub(rx, '', effe)

        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(word_tokenize(line)):
            index = line.find(w, index)

            if not index == -1:
                d[idx].append([w, index])
                #print(w, index)
                index += len(w)

        d_ = defaultdict(list)
        for idx in d:
            d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])

        init_e = line.find(effe)
        init_c = line.find(caus)

        for c, cl in enumerate(word_tokenize(caus)):
            print('init_c', init_c)
            init_c = line.find(cl, init_c)
            print('start Cause', init_c)
            stop = line.find(cl, init_c) + len(cl)
            word = line[init_c:stop]
            print('word', word.upper(), 'el', cl.upper())

            for idx in d_:
                if int(init_c) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([cl, 'C']), line.find(cl, init_c)])
                    d_[idx] = und_[idx]

            init_c += len(cl)
            print('increment_c', init_c)

        for e, el in enumerate(word_tokenize(effe)):
            print('init_e', init_e)
            init_e = line.find(el, init_e)
            print('start Effect', init_e)
            stop = line.find(el, init_e) + len(el)
            word = line[init_e:stop]
            print('word', word.upper(), 'el', el.upper())

            for idx in d_:
                if int(init_e) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'E']), line.find(word, init_e)])
                    d_[idx] = und_[idx]

            init_e += len(word)
            print('init_e', init_e)

        dd[i].append(d_)

        for dict_ in dd:
            dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])

    return dd_

def nltkPOS(loft):

    su_pos = []
    rx = re.compile(r"(\b[-']\b)|[\W_]")
    rxlist = [r'("\\)', r'(\\")']
    rx = re.compile('|'.join(rxlist))

    for i, j in enumerate(loft):
        text = re.sub(rx, '', j)
        tokens = word_tokenize(text)
        pos_ = list(nltk.pos_tag(tokens))
        su_pos.append(pos_)

    return su_pos



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
    """
    :param doc:
    :return:
    """
    return [word2features(doc, i) for i in range(len(doc))]


# A function fo generating the list of labels for each document: TOKEN, POS, LABEL
def get_multi_labels(doc):
    """
    :param doc:
    :return:
    """
    return [label for (token, postag, label) in doc]





if __name__ == '__main__':

    import pandas as pd

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
    postags = nltkPOS([i['sentence'] for i in lodict_])

    for i, (j, k) in enumerate(zip(hometags, postags)):
        if len(j) != len(k):
            print('POS alignement warning, ', i)
        else:
            print('Sizing OK')



