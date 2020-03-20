
import argparse
from utils import *
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pycrfsuite
import os
import pickle
import nltk
text = "learn php from guru99"
tokens = nltk.word_tokenize(text)
print(tokens)
tag = nltk.pos_tag(tokens)
print(tag)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inrepo', type = str, help = 'input repo')
    parser.add_argument('--outrepo', type = str, help = 'output repo')
    parser.add_argument('--idx', type = str, help = 'experience index')



    args = parser.parse_args()

    #df = pd.read_csv(args.repo, delimiter=';', header=0)
    df = pd.read_csv("./data/fnp2020-fincausal2-task2.csv", delimiter=';', header=0)


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
    sent = [i['sentence'] for i in lodict_]

    data = []
    for i, (j, k) in enumerate(zip(hometags, postags)):
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(j, k)])

    X = [extract_features(doc) for doc in data]
    y = [get_multi_labels(doc) for doc in data]

    # ####################  MAKE TRAIN TEST SETS ############################
    from sklearn.utils import shuffle

    size = 0.2
    seed = 42
    n = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    X_train_sent, X_test_sent = train_test_split(sent, test_size=size, random_state=seed)

    trainer = pycrfsuite.Trainer(verbose=True)

    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # trainer.select("l2sgd")
    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,
        'feature.minfreq': 0.7,

        'feature.possible_states': True,
        # coefficient for L2 penalty
        'c2': 0.01,

        # maximum number of iterations
        'max_iterations': "3000",

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })

    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    if not os.path.exists('models'):
        os.makedirs("models")
    writepath_ = os.path.join("models", str(args.idx))
    if not os.path.exists(writepath_):
        os.makedirs(writepath_)
    trainer.train(os.path.join(writepath_, ("crf_" + str(args.idx)) + ".model"))

    data_list = [X_train, X_test, y_train, y_test]
    pickle.dump(data_list, open(os.path.join(writepath_, ("data_" + str(args.idx)) + ".dat"), "wb"))

    tagger = pycrfsuite.Tagger()
    tagger.open(os.path.join(writepath_, ("crf_" + str(args.idx)) + ".model"))

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    labels = {"C": 1, "E": 2, "_": 0}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    print(np.sum(truths == predictions) / len(truths))


    # # Print out the classification report
    print(classification_report(
        truths, predictions,
        target_names=["C", "E", "_"]))

    # # Print out task2 metrics
    print(precision_recall_fscore_support(truths, predictions, average='weighted'))

    # print results and make tagged sentences
    ll = []
    for i in range(len(X_test) - 1):
        l = defaultdict(list)
        for j, (y, x) in enumerate(zip(y_pred[i], list(zip(*[[v for k, v in x.items()] for x in X_test[i]]))[1])):
            # print("%s (%s)" % (y, x))
            l.update({x: y})
        ll.append(l)

    nl = []
    for line, yt, yp, s in zip(ll, y_test, y_pred, X_test_sent):
        # print(line)
        d_ = defaultdict(list)
        d_["origin"] = s
        d_["truth"] = yt
        d_["pred"] = yp
        d_["diverge"] = 0
        for k, v in line.items():
            d_[v].append(k)
        if d_["truth"] != d_["pred"]:
            d_["diverge"] = 1
        nl.append(d_)

    print(sum([i["diverge"] for i in nl if i["diverge"] == 1]))

    fieldn = sorted(list(set(k for d in nl for k in d)))
    # with open('models/test_crf_1.csv', "w+") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldn, delimiter="~")
    #     writer.writeheader()
    #     for line in nl:
    #         writer.writerow(line)


    trainer.logparser.last_iteration

    info = tagger.info()


    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-15:])


    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))


    print("Top positive:")
    print_state_features(Counter(info.state_features).most_common(20))

    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-20:])

