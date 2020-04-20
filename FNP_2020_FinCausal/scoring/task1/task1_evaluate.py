#!/usr/bin/env python
# coding=utf-8
""" task1_evaluate.py - Scoring program for Fincausal 2020 Task 1

    usage: task1_evaluate.py [-h] {from-folder,from-file} ...

    positional arguments:
      {from-folder,from-file}
                            Use from-file for basic mode or from-folder for
                            Codalab compatible mode

Usage 1: Folder mode

    usage: task1_evaluate.py from-folder [-h] input output

    Codalab mode with input and output folders

    positional arguments:
      input       input folder with ref (reference) and res (result) sub folders
      output      output folder where score.txt is written

    optional arguments:
      -h, --help  show this help message and exit
    task1_evaluate input output

    input, output folders must follow the Codalab competition convention for scoring bundle
    e.g.
        ├───input
        │   ├───ref
        │   └───res
        └───output

Usage 2: File mode

    usage: usage: task1_evaluate.py from-file [-h] [--ref_file REF_FILE] pred_file [score_file]

    Basic mode with path to input and output files

    positional arguments:
      ref_file    reference file (default: ../../data/fnp2020-fincausal-task1.csv)
      pred_file   prediction file to evaluate
      score_file  path to output score file (or stdout if not provided)

    optional arguments:
      -h, --help  show this help message and exit
"""

import argparse
import logging
import sys
import os
import unittest

from sklearn import metrics


def get_values(csv_lines):
    """
    Retrieve labels from CSV content. Separator must be ';' and label is the last field.
    :param csv_lines:
    :return: list of labels
    """
    return [int(line.split(';')[-1]) for line in csv_lines]


def evaluate(gold_file, submission_file, output_file=None):
    """
    Evaluate Precision, Recall, F1 scores between gold_file and submission_file
    If output_file is provided, scores are saved in this file otherwise printed to std output.
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

        # get values (skipping headers)
        logging.info('* Loading reference data')
        y_true = get_values(ref_csv[1:])
        logging.info('* Loading prediction data')
        y_pred = get_values(sub_csv[1:])

        logging.info(f'Load Data: check data set length = {len(y_true) == len(y_pred)}')
        assert len(y_true) == len(y_pred)

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                           labels=[0, 1], average='weighted')

        scores = [
            "F1: %f\n" % f1,
            "Recall: %f\n" % recall,
            "Precision: %f\n" % precision,
            "ExactMatch: %f\n" % -1.0
        ]

        for s in scores:
            print(s, end='')

        if output_file is not None:
            with open(output_file, 'w', encoding='utf-8') as fp:
                for s in scores:
                    fp.write(s)


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

        evaluate(g_file, s_file, o_file)


def from_file(args):
    return evaluate(args.ref_file, args.pred_file, args.score_file)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Use from-file for basic mode or from-folder for Codalab compatible mode')

    command1_parser = subparsers.add_parser('from-folder', description='Codalab mode with input and output folders')
    command1_parser.set_defaults(func=from_folder)
    command1_parser.add_argument('input', help='input folder with ref (reference) and res (result) sub folders')
    command1_parser.add_argument('output', help='output folder where score.txt is written')

    command2_parser = subparsers.add_parser('from-file', description='Basic mode with path to input and output files')
    command2_parser.set_defaults(func=from_file)
    command2_parser.add_argument('--ref_file', default='../../data/fnp2020-fincausal-task1.csv', help='reference file')
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
