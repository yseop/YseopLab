#!/usr/bin/env python
# coding=utf-8
""" task2_evaluate.py - Scoring program for Fincausal 2020 Task 2

    usage: task2_evaluate.py [-h] {from-folder,from-file} ...

    positional arguments:
      {from-folder,from-file}
                            Use from-file for basic mode or from-folder for
                            Codalab compatible mode

Usage 1: Folder mode

    usage: task2_evaluate.py from-folder [-h] input output

    Codalab mode with input and output folders

    positional arguments:
      input       input folder with ref (reference) and res (result) sub folders
      output      output folder where score.txt is written

    optional arguments:
      -h, --help  show this help message and exit
    task2_evaluate input output

    input, output folders must follow the Codalab competition convention for scoring bundle
    e.g.
        ├───input
        │   ├───ref
        │   └───res
        └───output

Usage 2: File mode

    usage: task2_evaluate.py from-file [-h] [--ref_file REF_FILE] pred_file [score_file]

    Basic mode with path to input and output files

    positional arguments:
      ref_file    reference file (default: ../../data/fnp2020-fincausal-task2.csv)
      pred_file   prediction file to evaluate
      score_file  path to output score file (or stdout if not provided)

    optional arguments:
      -h, --help  show this help message and exit
"""
import argparse
import logging
import os
import unittest

from collections import namedtuple

import nltk

from sklearn import metrics


def build_token_index(text):
    """
    build a dictionary of all tokenized items from text with their respective positions in the text.
    E.g. "this is a basic example of a basic method" returns
     {'this': [0], 'is': [1], 'a': [2, 6], 'basic': [3, 7], 'example': [4], 'of': [5], 'method': [8]}
    :param text: reference text to index
    :return: dict() of text token, each token with their respective position(s) in the text
    """
    tokens = nltk.word_tokenize(text)
    token_index = {}
    for position, token in enumerate(tokens):
        if token in token_index:
            token_index[token].append(position)
        else:
            token_index[token] = [position]
    return tokens, token_index


def get_tokens_sequence(text, token_index):
    tokens = nltk.word_tokenize(text)
    # build list of possible position for each token
    # positions = [word_index[word] for word in words]
    positions = []
    for token in tokens:
        if token in token_index:
            positions.append(token_index[token])
            continue
        # Special case when '.' is not tokenized properly
        alt_token = ''.join([token, '.'])
        if alt_token in token_index:
            logging.debug(f'tokenize fix ".": {alt_token}')
            positions.append(token_index[alt_token])
            # TODO: discard the next token if == '.' ?
            continue
        # Special case when/if ',' is not tokenized properly - TBC
        # alt_token = ''.join([token, ','])
        # if alt_token in token_index:
        #     logging.debug(f'tokenize fix ",": {alt_token}')
        #     positions.append(token_index[alt_token])
        #     continue
        else:
            logging.warning(f'get_tokens_sequence "{token}" discarded')
    # No matching ? stop here
    if len(positions) == 0:
        return positions
    # recursively process the list of token positions to return combinations of consecutive tokens
    seqs = _get_sequences(*positions)
    # Note: several sequences can possibly be found in the reference text, when similar text patterns are repeated
    # always return the longest
    return max(seqs, key=len)


def _get_sequences(*args, value=None, path=None):
    """
    Recursive method to select sequences of successive tokens using their position relative to the reference text.
    A sequence is the list of successive indexes in the tokenized reference text.
    Implemented as a product() of successive positions constrained by their
    :param args: list of list of positions
    :param value: position of the previous token (i.e. next token position must be in range [value+1, value+3]
    :param path: debugging - current sequence
    :return:
    """
    logging.debug(path)
    # end of recursion
    if len(args) == 1:
        if value is not None:
            # return items matching constraint (i.e. within range with previous token)
            return [x for x in args[0] if x > value and (x < value+3)]
        else:
            # Special case where text is restricted to a single token
            # return all positions on first call (i.e. value is None)
            return [args[0]]
    else:
        # iterate over current token possible positions and combine with other tokens from recursive call
        # result is a list of explored sequences (i.e. list of list of positions)
        result = []
        for x in args[0]:
            # <Debug> keep track of current explored sequence
            p = [x] if path is None else list(path + [x])
            # </Debug>
            if value is None or (x > value and (x < value+3)):
                seqs = _get_sequences(*args[1:], value=x, path=p)
                # when recursion returns empty list and current position match constraint (either only value
                # or value within range) add current position as a single result
                if len(seqs) == 0 and (value is None or (x > value and (x < value+3))):
                    result.append([x])
                else:
                    # otherwise combine current position with recursion results (whether returned sequences are list
                    # or single number) and add to the list of results for this token position
                    for s in seqs:
                        res = [x] + s if type(s) is list else [x, s]
                        result.append(res)
        return result


def encode_causal_tokens(text, cause, effect):
    """
    Encode text, cause and effect into a single list with each token represented by their respective
    class labels ('-','C','E')
    :param text: reference text
    :param cause: causal substring in reference text
    :param effect: effect substring in reference text
    :return: text string converted as a list of tuple(token, label)
    """
    # Get reference text tokens and token index
    logging.debug(f'Reference: {text}')
    words, wi = build_token_index(text)
    logging.debug(f'Token index: {wi}')

    # init labels with default class label
    labels = ['-' for _ in range(len(words))]

    # encode cause using token index
    logging.debug(f'Cause: {cause}')
    cause_seq = get_tokens_sequence(cause, wi)
    logging.debug(f'Cause seq.: {cause_seq}')
    for position in cause_seq:
        labels[position] = 'C'

    # encode effect using token index
    logging.debug(f'Effect: {effect}')
    effect_seq = get_tokens_sequence(effect, wi)
    logging.debug(f'Effect seq.: {effect_seq}')
    for position in effect_seq:
        labels[position] = 'E'

    logging.debug(labels)

    return zip(words, labels)


def evaluate(truth, predict, classes):
    """
    Fincausal 2020 Task 2 evaluation: returns precision, recall and F1 comparing submitting data to reference data.
    :param truth: list of Task2Data(index, text, cause, effect, labels) - reference data set
    :param predict: list of Task2Data(index, text, cause, effect, labels) - submission data set
    :param classes: list of classes
    :return: tuple(precision, recall, f1, exact match)
    """
    exact_match = 0
    y_truth = []
    y_predict = []
    multi = {}
    # First pass - process text sections with single causal relations and store others in `multi` dict()
    for t, p in zip(truth, predict):
        # Process Exact Match
        exact_match += 1 if all([x == y for x, y in zip(t.labels, p.labels)]) else 0
        # PRF: Text section with multiple causal relationship ?
        if t.index.count('.') == 2:
            # extract root index and add to the list to be processed later
            root_index = '.'.join(t.index.split('.')[:-1])
            if root_index in multi:
                multi[root_index][0].append(t.labels)
                multi[root_index][1].append(p.labels)
            else:
                multi[root_index] = [[t.labels], [p.labels]]
        else:
            # Accumulate data for precision, recall, f1 scores
            y_truth.extend(t.labels)
            y_predict.extend(p.labels)
    # Second pass - deal with text sections having multiple causal relations
    for index, section in multi.items():
        # section[0] list of possible truth labels
        # section[1] list of predicted labels
        candidates = section[1]
        # for each possible combination of truth labels - try to find the best match in predicted labels
        # then repeat, removing this match from the list of remaining predicted labels
        for t in section[0]:
            best = None
            for p in candidates:
                f1 = metrics.f1_score(t, p, labels=classes, average='weighted', zero_division=0)
                if best is None or f1 > best[1]:
                    best = (p, f1)
            # Use best to add to global evaluation
            y_truth.extend(t)
            y_predict.extend(best[0])
            # Remove best from list of candidate for next iteration
            candidates.remove(best[0])
        # Ensure all candidate predictions have been reviewed
        assert len(candidates) == 0

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_truth, y_predict,
                                                                       labels=classes,
                                                                       average='weighted',
                                                                       zero_division=0)

    import numpy as np
    """
    Sklearn Multiclass confusion matrix is:
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    cmc = multilabel_confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    cmc
    array([[[3, 1],
            [0, 2]],

           [[5, 0],
            [1, 0]],

           [[2, 1],
            [1, 2]]])

    cm_ant = cmc[0]
    tp = cm_ant[1,1]	has been classified as ant and is ant
    fp = cm_ant[0,1]	has been classified as ant and is sth else
    fn = cm_ant[1,0]    has been classified as sth else than ant and is ant
    tn = cm_ant[0,0]    has been classified as sth else than ant and is something else

    """
    print(' ')
    print('raw metrics')
    print("#-------------------------------------------------------------------------------------#")

    MCM = metrics.multilabel_confusion_matrix(y_truth, y_predict, labels=classes)
    tp_sum = MCM[:, 1, 1]
    print('tp_sum ', tp_sum)
    pred_sum = tp_sum + MCM[:, 0, 1]
    print('predicted in the classes ', pred_sum)
    true_sum = tp_sum + MCM[:, 1, 0]
    print('actually in the classes (tp + fn: support) ', true_sum)

    """beta : float, 1.0 by default
    The strength of recall versus precision in the F-score."""

    precision_ = tp_sum / pred_sum
    recall_ = tp_sum / true_sum
    beta = 1.0
    beta2 = beta ** 2
    denom = beta2 * precision_ + recall_
    print('denom', denom)
    denom[denom == 0.] = 1  # avoid division by 0
    weights = true_sum
    print("weights", weights)
    print(' ')
    print("#-------------------------------------------------------------------------------------#")
    print('macro scores')
    f_score_ = (1 + beta2) * precision_ * recall_ / denom
    print('macro precision ', sum(precision_) / 3, 'macro recall', sum(recall_) / 3, 'macro f_score ',
          sum(f_score_) / 3)

    ## recompute for average from source

    precision_ = np.average(precision_, weights=weights)
    recall_ = np.average(recall_, weights=weights)
    f_score_ = np.average(f_score_, weights=weights)
    print(' ')
    print("#-------------------------------------------------------------------------------------#")
    print('recomputed weighted metrics ')
    print('weighted precision', precision_, 'weighted recall', recall_, 'weighted_fscore', f_score_)
    print(' ')
    print("#-------------------------------------------------------------------------------------#")
    print('classification report')
    print(metrics.classification_report(y_truth, y_predict, target_names=classes))

    logging.debug(f'SKLEARN EVAL: {f1}, {precision}, {recall}')

    return precision, recall, f1, float(exact_match)/float(len(truth))


Task2Data = namedtuple('Task2Data', ['index', 'text', 'cause', 'effect', 'labels'])


def get_data(csv_lines):
    """
    Retrieve Task 2 data from CSV content (separator is ';') as a list of (index, text, cause, effect).
    :param csv_lines:
    :return: list of Task2Data(index, text, cause, effect, labels)
    """
    result = []
    for line in csv_lines:
        line = line.rstrip('\n')

        index, text, cause, effect = line.split(';')[:4]

        text = text.lstrip()
        cause = cause.lstrip()
        effect = effect.lstrip()

        _, labels = zip(*encode_causal_tokens(text, cause, effect))

        result.append(Task2Data(index, text, cause, effect, labels))

    return result


def evaluate_files(gold_file, submission_file, output_file=None):
    """
    Evaluate Precision, Recall, F1 scores between gold_file and submission_file
    If output_file is provided, scores are saved in this file and printed to std output.
    :param gold_file: path to reference data
    :param submission_file: path to submitted data
    :param output_file: path to output file as expected by Codalab competition framework
    :return:
    """
    if os.path.exists(gold_file) and os.path.exists(submission_file):
        with open(gold_file, 'r', encoding='utf-8') as fp:
            ref_csv = fp.readlines()
        with open(submission_file, 'r', encoding='utf-8') as fp:
            sub_csv = fp.readlines()

        # Get data (skipping headers)
        logging.info('* Loading reference data')
        y_true = get_data(ref_csv[1:])
        logging.info('* Loading prediction data')
        y_pred = get_data(sub_csv[1:])

        logging.info(f'Load Data: check data set length = {len(y_true) == len(y_pred)}')
        logging.info(f'Load Data: check data set ref. text = {all([x.text == y.text for x, y in zip(y_true, y_pred)])}')
        assert len(y_true) == len(y_pred)
        assert all([x.text == y.text for x, y in zip(y_true, y_pred)])

        # Process data using classes: -, C & E
        precision, recall, f1, exact_match = evaluate(y_true, y_pred, ['-', 'C', 'E'])

        scores = [
            "F1: %f\n" % f1,
            "Recall: %f\n" % recall,
            "Precision: %f\n" % precision,
            "ExactMatch: %f\n" % exact_match
        ]

        for s in scores:
            print(s, end='')
        if output_file is not None:
            with open(output_file, 'w', encoding='utf-8') as fp:
                for s in scores:
                    fp.write(s)
    else:
        # Submission file most likely being the wrong one - tell which one we are looking for
        logging.error(f'{os.path.basename(gold_file)} not found')

    ## Save for control
    import pandas as pd
    df = pd.DataFrame.from_records(y_true)
    df.columns = ['Index', 'Text', 'Cause', 'Effect', 'TRUTH']
    dfpred = pd.DataFrame.from_records(y_pred)
    dfpred.columns = ['Index', 'Text', 'Cause', 'Effect', 'PRED']
    df['PRED'] = dfpred['PRED']
    df['TRUTH'] = df['TRUTH'].apply(lambda x: ' '.join(x))
    df['PRED'] = df['PRED'].apply(lambda x: ' '.join(x))


    ctrlpath = submission_file.split('/')
    ctrlpath.pop()
    ctrlpath = '/'.join([path_ for path_ in ctrlpath])
    df.to_csv(os.path.join(ctrlpath, 'origin_control.csv'), header=1, index=0)


def from_folder(args):
    # Folder mode - Codalab usage
    submit_dir = os.path.join(args.input, 'res')
    truth_dir = os.path.join(args.input, 'ref')
    output_dir = args.output

    if not os.path.isdir(submit_dir):
        logging.error("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    o_file = os.path.join(output_dir, 'scores.txt')

    gold_list = os.listdir(truth_dir)
    for gold in gold_list:
        g_file = os.path.join(truth_dir, gold)
        s_file = os.path.join(submit_dir, gold)

        evaluate_files(g_file, s_file, o_file)


def from_file(args):
    return evaluate_files(args.ref_file, args.pred_file, args.score_file)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Use from-file for basic mode or from-folder for Codalab compatible mode')

    command1_parser = subparsers.add_parser('from-folder', description='Codalab mode with input and output folders')
    command1_parser.set_defaults(func=from_folder)
    command1_parser.add_argument('input', help='input folder with ref (reference) and res (result) sub folders')
    command1_parser.add_argument('output', help='output folder where score.txt is written')

    command2_parser = subparsers.add_parser('from-file', description='Basic mode with path to input and output files')
    command2_parser.set_defaults(func=from_file)
    command2_parser.add_argument('--ref_file', default='../../data/fnp2020-fincausal-task2.csv', help='reference file')
    command2_parser.add_argument('pred_file', help='prediction file to evaluate')
    command2_parser.add_argument('score_file', nargs='?', default=None,
                                 help='path to output score file (or stdout if not provided)')

    logging.basicConfig(level=logging.INFO,
                        filename=None,
                        format='%(levelname)-7s| %(message)s')

    args = parser.parse_args()
    if 'func' in args:
        exit(args.func(args))
    else:
        parser.print_usage()
        exit(1)


if __name__ == '__main__':
    main()


# Tests, which can be executed with `python -m unittest task2_evaluate`.
class Test(unittest.TestCase):
    def _process_test_ok(self, t_text, p_text, t_labels, p_labels, f1, precision, recall, exact_match):
        # Load data
        y_true = get_data(t_text)
        y_pred = get_data(p_text)
        with self.subTest(value='encode_causal_truth'):
            self.assertEqual(y_true[0].labels, t_labels)
        with self.subTest(value='encode_causal_pred'):
            self.assertEqual(y_pred[0].labels, p_labels)
        # Evaluate precision, recall, f1 and exact matches
        result = evaluate(y_true, y_pred, ['-', 'C', 'E'])
        # Round result to 2 decimals
        result = tuple(map(lambda x: round(x, 2), result))
        with self.subTest(value='evaluate'):
            self.assertEqual(result, (precision, recall, f1, exact_match))

    def test_0(self):
        """ Identity """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj.; aaa bbbb; hhh i jjjj.\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj.; aaa bbbb; hhh i jjjj.\n'],
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E', 'E'),
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E', 'E'),
            1.00, 1.00, 1.00, 1.00
        )

    def test_1(self):
        """ single failure """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa bbbb; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb; hhh i jjjj\n'],
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E'),
            ('-', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E'),
            0.92, 0.9, 0.89, 0.0
        )

    def test_2(self):
        """ 1 missed 1 error """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa bbbb; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb; gggg hhh i jjjj\n'],
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E'),
            ('-', 'C', '-', '-', '-', '-', 'E', 'E', 'E', 'E'),
            0.82, 0.8, 0.79, 0.0
        )

    def test_3(self):
        """ total failure """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj.; aaa bbbb; hhh i jjjj.\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj.; ccc d; eeee ffff gggg\n'],
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E', 'E'),
            ('-', '-', 'C', 'C', 'E', 'E', 'E', '-', '-', '-', '-'),
            0.0, 0.0, 0.0, 0.0
        )

    def test_4(self):
        """ punctuation """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj.; aaa bbbb; hhh i jjjj.\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj.; aaa bbbb; hhh i jjjj\n'],
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E', 'E'),
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E', '-'),
            0.92, 0.91, 0.91, 0.0
        )

    def test_5(self):
        """ 2 missed + punctuation """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d aaa bbbb ccc hhh i jjjj.; aaa bbbb ccc d; hhh i jjjj.\n'],
            ['1.0; aaa bbbb ccc d aaa bbbb ccc hhh i jjjj.; aaa bbbb; hhh i jjjj\n'],
            ('C', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E', 'E'),
            ('C', 'C', '-', '-', '-', '-', '-', 'E', 'E', 'E', '-'),
            0.86, 0.73, 0.74, 0.0
        )

    def test_6(self):
        """ non consecutive tokens (out of range) """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa d; hhh i jjjj\n'],
            ('C', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('C', '-', '-', '-', '-', '-', '-', 'E', 'E', 'E'),
            0.85, 0.7, 0.66, 0.0
        )

    def test_7(self):
        """ non consecutive tokens (in range) """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb d; hhh i jjjj\n'],
            ('C', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('-', 'C', '-', 'C', '-', '-', '-', 'E', 'E', 'E'),
            0.88, 0.8, 0.79, 0.0
        )

    def test_8(self):
        """ no effect """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d;\n'],
            ('-', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('-', 'C', 'C', 'C', '-', '-', '-', '-', '-', '-'),
            0.53, 0.7, 0.59, 0.0
        )

    def test_9(self):
        """ no cause """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; ; bbbb ccc d\n'],
            ('-', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('-', 'E', 'E', 'E', '-', '-', '-', '-', '-', '-'),
            0.23, 0.4, 0.29, 0.0
        )

    def test_10(self):
        """ no cause, no effect """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; ; \n'],
            ('-', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('-', '-', '-', '-', '-', '-', '-', '-', '-', '-'),
            0.16, 0.4, 0.23, 0.0
        )

    def test_11(self):
        """ all cause """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; \n'],
            ('-', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'),
            0.09, 0.3, 0.14, 0.0
        )

    def test_12(self):
        """ half cause  """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; aaa ccc eeee gggg i;\n'],
            ('-', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('C', '-', 'C', '-', 'C', '-', 'C', '-', 'C', '-'),
            0.14, 0.2, 0.16, 0.0
        )

    def test_13(self):
        """ all effect """
        self._process_test_ok(
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; bbbb ccc d; hhh i jjjj\n'],
            ['1.0; aaa bbbb ccc d eeee ffff gggg hhh i jjjj; ; aaa bbbb ccc d eeee ffff gggg hhh i jjjj\n'],
            ('-', 'C', 'C', 'C', '-', '-', '-', 'E', 'E', 'E'),
            ('E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'),
            0.09, 0.3, 0.14, 0.0
        )
