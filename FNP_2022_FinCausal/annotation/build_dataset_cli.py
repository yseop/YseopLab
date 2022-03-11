""" fnp2022 build-dataset command implementation
<REFACTORING IN-PROGRESS>
## BRAT configuration detailed below:
**annotation.conf**
```
# Simple annotation scheme for FinCausal
[entities]
QFact
Fact
Discard
Remove
Review
Quantity
Done
[relations]
Cause 	   Cause:Fact, Effect:QFact
Cause	   Cause:QFact, Effect:QFact
<OVERLAP> Arg1:Quantity, Arg2:QFact, <OVL-TYPE>:contain
<OVERLAP> Arg1:Quantity, Arg2:Fact, <OVL-TYPE>:contain
[events]
[attributes]
```
"""

import csv
import logging
import os
import random
import re
import sys
from collections import namedtuple
import xlsxwriter
from tqdm import tqdm
import brat
from pathlib import Path
import glob
from argparse import ArgumentParser

class XLSXAnnotationDetails:
    """
    Wrapper for Annotation Details spreadsheet
    """
    re_tag_split = re.compile(r'(<e1>.*</e1>|<e2>.*</e2>)')

    def __init__(self, filename):
        """
        Create a new Annotation Details spreadsheet wrapper
        :param filename: path to file
        """
        self._workbook = xlsxwriter.Workbook(filename)

        self._init_formats()

        self._init_info_sheet()
        self._init_data_sets_sheet()
        self._init_annotations_sheet()
        self._init_fact_sheet()

    def _init_formats(self):
        self._format_top_aligned = self._workbook.add_format({'valign': 'top'})
        self._format_top_aligned_center = self._workbook.add_format({'valign': 'top', 'align': 'center'})
        self._format_top_aligned_wrap = self._workbook.add_format({'valign': 'top'})
        self._format_top_aligned_wrap.set_text_wrap()
        self._format_top_aligned_url = self._workbook.get_default_url_format()

        self._format_top_aligned_url.set_align('top')
        self._format_top_aligned_url.set_align('center')
        self._format_cause = self._workbook.add_format({'bold': True, 'font_color': 'blue'})
        self._format_effect = self._workbook.add_format({'bold': True, 'font_color': 'red'})

    def _init_info_sheet(self):
        self._sheet_info = self._workbook.add_worksheet('Info')
        self._sheet_info_row = 0

    def info_print(self, data):
        if type(data) == list:
            self._sheet_info.write_row(self._sheet_info_row, 0, data)
        else:
            self._sheet_info.write(self._sheet_info_row, 0, data)

        self._sheet_info_row += 1

    def _init_data_sets_sheet(self):
        self._sheet_data_sets = self._workbook.add_worksheet('Data sets')
        self._sheet_data_sets_row = 1
        self._sheet_data_sets.write_row(0, 0,
                                        ['URL', 'Index', 'Text', 'Gold', 'Cause', 'Effect',
                                         'Offset_Sentence2', 'Offset_Sentence3',
                                         'Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End',
                                         'Tagged_Text'])
        # All Columns are top aligned - default width = 10 except for the first column = 5
        self._sheet_data_sets.set_column(0, 0, 5, cell_format=self._format_top_aligned)
        self._sheet_data_sets.set_column(1, 10, 10, cell_format=self._format_top_aligned)
        # + width = 50 and text wrapping for Text, Cause, Effect, Tagged Text
        self._sheet_data_sets.set_column(2, 2, 50, cell_format=self._format_top_aligned_wrap)
        self._sheet_data_sets.set_column(4, 5, 50, cell_format=self._format_top_aligned_wrap)
        self._sheet_data_sets.set_column(12, 12, 50, cell_format=self._format_top_aligned_wrap)
        self._sheet_data_sets.freeze_panes(1, 2)

    def _get_formatted_tagged_text(self, text):
        """
        Detect <e1>, <e2> tags in text and create text segments + formats to use with write_rich_string
        https://xlsxwriter.readthedocs.io/example_rich_strings.html
            Sample:
            segments = ['This is ', bold, 'bold', ' and this is ', blue, 'blue']
            worksheet.write_rich_string('A9', *segments)
        :param text:
        :return: list to use with write_rich_string
        """
        strings = re.split(self.re_tag_split, text)
        res = []
        for s in strings:
            if '<e1>' in s:
                res.append(self._format_cause)
            if '<e2>' in s:
                res.append(self._format_effect)
            if len(s) > 0:
                res.append(s)
        return res

    def add_data_sets(self, url, index, text, gold_label, cause='', effect='',
                      cause_start='', cause_end='', effect_start='', effect_end='', tagged_text='',
                      sentence_offsets=None):
        self._sheet_data_sets.write_url(self._sheet_data_sets_row, 0, url,
                                        string='Link',
                                        cell_format=self._format_top_aligned_url)
        self._sheet_data_sets.write_row(self._sheet_data_sets_row, 1, [index, text, gold_label, cause, effect])
        if len(sentence_offsets) > 0:
            self._sheet_data_sets.write_row(self._sheet_data_sets_row, 6, sentence_offsets)
        self._sheet_data_sets.write_row(self._sheet_data_sets_row, 8,
                                        [cause_start, cause_end, effect_start, effect_end])

        formatted_string = self._get_formatted_tagged_text(tagged_text)
        if len(formatted_string) > 1:
            self._sheet_data_sets.write_rich_string(self._sheet_data_sets_row, 12, *formatted_string)
        else:
            self._sheet_data_sets.write(self._sheet_data_sets_row, 12, tagged_text)

        self._sheet_data_sets_row += 1

    def _init_annotations_sheet(self):
        self._sheet_annotations = self._workbook.add_worksheet('Annotations')
        self._sheet_annotations_row = 1
        self.annotations = ['Done', 'BlindDone', 'Verified', 'Extra', 'Cause', 'Fact', 'QFact', 'Effect', 'Quantity',
                            'Discard', 'Remove', 'Review', 'CausalCandidate', 'AnnotatorNotes', 'Notes']
        col_first_annotation = 2
        col_last_annotation = col_first_annotation + len(self.annotations) - 1

        self._sheet_annotations.write_row(0, 0, ['Url', 'Lines'])
        self._sheet_annotations.write_row(0, col_first_annotation, self.annotations)

        # All Columns are top aligned - default width = 10 except for the first column = 30
        self._sheet_annotations.set_column(0, 0, 30, cell_format=self._format_top_aligned)
        self._sheet_annotations.set_column(1, col_last_annotation - 1, 10, cell_format=self._format_top_aligned)
        # + width = 100 and text wrapping for last column (Notes)
        self._sheet_annotations.set_column(col_last_annotation, col_last_annotation, 100,
                                           cell_format=self._format_top_aligned_wrap)
        self._sheet_annotations.freeze_panes(1, 2)

    def _parse_annotation_file(self, filename):
        result = dict()
        # Init count for each annotation in result dictionary
        for key in self.annotations[:-1]:
            result[key] = 0
        # Assume last from the list is for notes/comments = string
        result['Notes'] = ''

        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                try:
                    ann_type = line.split('\t')[1].split()[0]
                except IndexError:
                    tqdm.write(f'{filename} parsing error: {sys.exc_info()[1]}, {repr(line)}')
                    continue
                if ann_type in result:
                    result[ann_type] += 1
                else:
                    tqdm.write(f'{filename} unknown annotation: {repr(line)}')
                if ann_type == 'AnnotatorNotes':
                    comment = line.split('\t')[2]
                    if len(result['Notes']):
                        result['Notes'] = f'{result["Notes"]}, {comment}'
                    else:
                        result['Notes'] = comment
        return result

    def add_annotation_data(self, url, url_text, lines, filename):
        self._sheet_annotations.write_url(self._sheet_annotations_row, 0, url,
                                          string=url_text,
                                          cell_format=self._format_top_aligned_url)
        self._sheet_annotations.write(self._sheet_annotations_row, 1, lines)

        annotations_dict = self._parse_annotation_file(filename)
        for key, value in annotations_dict.items():
            column = self.annotations.index(key) + 2
            # Do not write 0 values
            if type(value) == int and value == 0:
                continue
            # Remove any newline
            if type(value) == str:
                value = value.replace('\n', '')

            self._sheet_annotations.write(self._sheet_annotations_row, column, value)

        self._sheet_annotations_row += 1

    def _init_fact_sheet(self):
        self._sheet_facts = self._workbook.add_worksheet('Fact Alignment')
        self._sheet_facts_row = 1
        self._sheet_facts.write_row(0, 0,
                                    ['URL', 'Index', 'Text', 'Sentence', 'Left', 'Right',
                                     'Aligned_Fact', 'Remain_Left', 'Remain_Right'])
        # First column is top aligned and width = 5
        self._sheet_facts.set_column(0, 0, 5, cell_format=self._format_top_aligned)
        # Index column is top aligned and width = 10
        self._sheet_facts.set_column(1, 1, 10, cell_format=self._format_top_aligned)
        # + width = 50 and text wrapping for Text, Left, Right, Aligned_Result,
        self._sheet_facts.set_column(2, 8, 50, cell_format=self._format_top_aligned_wrap)
        self._sheet_facts.freeze_panes(1, 2)

    def add_fact_alignment(self, url, index, text, sentence, left, right, aligned_text,
                           new_left, new_right):
        self._sheet_facts.write_url(self._sheet_facts_row, 0, url,
                                    string='Link',
                                    cell_format=self._format_top_aligned_url)
        self._sheet_facts.write(self._sheet_facts_row, 1, index)

        self._sheet_facts.write(self._sheet_facts_row, 2, text)

        formatted_string = self._get_formatted_tagged_text(sentence)
        if len(formatted_string) > 1:
            self._sheet_facts.write_rich_string(self._sheet_facts_row, 3, *formatted_string)
        else:
            self._sheet_facts.write(self._sheet_facts_row, 3, sentence)

        self._sheet_facts.write_row(self._sheet_facts_row, 4, [left, right])

        formatted_string = self._get_formatted_tagged_text(aligned_text)
        if len(formatted_string) > 1:
            self._sheet_facts.write_rich_string(self._sheet_facts_row, 6, *formatted_string)
        else:
            self._sheet_facts.write(self._sheet_facts_row, 6, text)

        self._sheet_facts.write_row(self._sheet_facts_row, 7, [new_left, new_right])

        self._sheet_facts_row += 1

    def close(self):
        self._sheet_data_sets.autofilter(0, 0, self._sheet_data_sets_row, 10)
        self._sheet_annotations.autofilter(0, 0, self._sheet_annotations_row, 12)
        self._sheet_facts.autofilter(0, 0, self._sheet_facts_row, 8)

        self._workbook.close()


def fnp2020_index(doc, index, sub_index=0):
    """
    Text indexing in task 1 and 2: sentences share the same index for both tasks, made of 2 or 3 parts separated with '.'.
    From left to right: the first 4-digit part padded with zeroes, represent the Id of the source document,
    the next 5-digit part padded with zeroes is the index of the sentence in that document.
    The optional third part made of a single digit is for task 2 when multiple causal relationship
    are present in the same sentence.
    """
    if sub_index == 0:
        return f'{doc:0>4}.{index:0>5}'
    else:
        return f'{doc:0>4}.{index:0>5}.{sub_index}'


def main(host, root, collection, doclist_file, output_dir):
    """
    """
    # Load the list of docs to process in the collection
    #docs = [os.path.join(root, collection, d.rstrip('\n')) for d in doclist_file]

    ann_files = [os.path.join(root, collection, Path(d).with_suffix('.ann')) for d in doclist_file]
    print(ann_files)
    print(root)
    print(collection)
    #######################################
    ### TODO: REFACTOR EVERYTHING BELOW ###
    #######################################

    INPUTDIR = root
    OUTPUTDIR = output_dir
    COLLECTION = collection
    SNAPSHOT = ''

    cleaning_re_list = []
    # "* Blockchain Basics *  First, we should ..." >> "First, we should ..."
    cleaning_re_list.append(re.compile(r'\*[\w :]+\*  '))
    # "___  BERNIE SANDERS: "We have the highest ..." >> "We have the highest ..."
    cleaning_re_list.append(re.compile(r'__[ \w]+: '))
    # "___  KAMALA HARRIS, on Trump: The only..." >> "KAMALA HARRIS, on Trump: The only..."
    cleaning_re_list.append(re.compile(r'^___  '))
    # " Posted by Jeanne O'Marion on Sep 25th, 2019  " << ""
    cleaning_re_list.append(re.compile(
        r" *Posted by [\w'\- ]+ on (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d+(st|nd|rd|th), \d+ +"))
    # "// Comments off  Kforce Inc." >> "."
    cleaning_re_list.append(re.compile(r'// Comments off[^.]+'))
    # Remove * followed by one or more spaces
    cleaning_re_list.append(re.compile(r'\* *'))

    def clean_text(text):
        for regexp in cleaning_re_list:
            text = re.sub(regexp, '', text)
        text = text.replace(';', '')
        text = text.replace('"', '')
        text = text.lstrip()
        return text

    if len(SNAPSHOT) == 0:
        collection_folder = COLLECTION
    else:
        collection_folder = '-'.join([COLLECTION, SNAPSHOT])
        OUTPUTDIR = os.path.join(OUTPUTDIR, SNAPSHOT)
    os.makedirs(OUTPUTDIR, exist_ok=True)

    fnp_task1_output_pre = os.path.join(OUTPUTDIR, f'fnp2022-{COLLECTION}-task1-pre.csv')
    fnp_task1_output = os.path.join(OUTPUTDIR, f'fnp2022-{COLLECTION}-task1.csv')
    fnp_task1_output_length = 0
    fnp_task1_output_distrib = [0, 0]
    fnp_task2_output_pre = os.path.join(OUTPUTDIR, f'fnp2022-{COLLECTION}-task2-pre.csv')
    fnp_task2_output = os.path.join(OUTPUTDIR, f'fnp2022-{COLLECTION}-task2.csv')
    fnp_task2_output_length = 0

    # TODO get rel path from drive


    def get_relative_path(filename):
        rel_path = filename.replace('.txt', '')
        rel_path = rel_path.replace(INPUTDIR, '')
        if len(SNAPSHOT):
            rel_path = rel_path.replace(f'-{SNAPSHOT}', '')
        rel_path = rel_path.replace('\\', '/')
        print('REL PATH', rel_path)
        return rel_path

    # Create reporting file
    xlsx_file = XLSXAnnotationDetails(os.path.join(OUTPUTDIR, 'fnp2022-annotation-details.xlsx'))

    # Pre-Processing - get statistics on annotated documents

    # 1- Retrieves the ratio of number of lines for blocs containing causal relationship
    # (Aim is to generate same ratio of number of lines for blocs without casual relationship)
    # Use a dict() where key is the line count and value is the count of occurrences
    # initializes values to 0 for the first 4 keys (bloc length should not exceed 3 lines but account for a few exceptions)
    random.seed(42)
    causal_lengths = [0 for x in range(4)]
    for ann_file in tqdm(ann_files, desc='Pre-Processing', leave=True):
        if os.path.getsize(ann_file) == 0:
            continue
        ann_file_txt = ann_file.replace('.ann', '.txt')
        af = brat.BratAnnotatedFile(ann_file_txt)
        if 'Cause' in af.annotation_types:
            # Retrieve all line intervals for 'Cause' binary relations
            intervals = [sorted([r.from_instance.lineno, r.to_instance.lineno])
                         for r in af.annotation_types['Cause'].values()
                         if isinstance(r, brat.BratAnnotationRelation)]
            for i in intervals:
                length = i[1] - i[0]
                if length < 4:
                    causal_lengths[length] += 1
                else:
                    tqdm.write(
                        f'ERROR: {get_relative_path(ann_file_txt)}: relation spanning on 3+ sentences ({length})')
        # Add annotations data to status spreadsheet
        doc = get_relative_path(ann_file_txt)
        print('DOC PATH', doc)
        # TODO add rel path to drive
        url = f'https://drive{doc}'
        lines = len(af.lines_text)
        xlsx_file.add_annotation_data(url, doc, lines, ann_file)

    total = sum(causal_lengths)
    try:
        causal_weights = [value / total for value in causal_lengths]
        # get rid of the last value (= 4 sentences)...
        causal_weights = causal_weights[:-1]
        # ...and ensure total distribution is 1.0
        causal_weights[-1] = 1.0 - sum(causal_weights[:-1])
        # Legitimate block lengths
        block_lengths = list(range(1, len(causal_weights) + 1))  # = [1, 2, 3] !
        # Target weights for non causal blocks
        target_weights = [v for v in causal_weights]
        # Initialize distribution of non causal block line length
        non_causal_lengths = [0 for _ in range(len(block_lengths))]
    except Exception as E:
        print(E)

    class LineData:
        """ Store data for a line of text including list of causal relationships
        """

        def __init__(self, index, text, causal_list):
            self.index = index
            self.text = text
            self._causal_list = causal_list

        def __repr__(self):
            return f'LineData({self.index}, {self.text}, {self._causal_list})'

        def add_causal_relation(self, relation):
            assert isinstance(relation, brat.BratAnnotationRelation)
            self._causal_list.append(relation)

        def has_causal_relation(self):
            return len(self._causal_list) > 0

        def causal_relations(self):
            return self._causal_list

    CausalData = namedtuple('CausalData', ['relation', 'start_index'])
    """ Helper structure to store causal relation with its start line index
    """

    def task1_print(index, text_block, gold_label):
        nonlocal writer1, fnp_task1_output_length
        out_text = clean_text(text_block)
        # keep track of sentence offsets
        sentence_offsets = [x for x, v in enumerate(out_text) if v == '\t']
        # get rid of sentence markers
        out_text = out_text.replace('\t', ' ')
        writer1.writerow([index, out_text, gold_label])
        fnp_task1_output_length += 1

        # HELPER: report spreadsheet: merge task 1 and task 2 output
        # Note: only when gold label is 0, as gold label 1 will be managed together with task 2 output
        nonlocal relative_path, line_data, ld_index
        if gold_label == '0':
            fnp_task1_output_distrib[0] += 1
            # TO CHECK: Why ld_index-1 vs ld_index when gold_label == 1 ?
            line = line_data[ld_index - 1].index + 1
        else:
            fnp_task1_output_distrib[1] += 1
            line = line_data[ld_index].index + 1
        # TODO add relative path to drive
        t1_url =f'https://drive{doc}?focus=sent~{line}'

        if gold_label == '0':
            xlsx_file.add_data_sets(t1_url, index, out_text, gold_label, sentence_offsets=sentence_offsets)

        if len(sentence_offsets) > 2:
            tqdm.write(f'WARNING: {relative_path}:{line} {len(sentence_offsets) + 1} sentences in text section')

    Task2ResultData = namedtuple('Task2ResultData',
                                 ['cause', 'effect', 'start_line', 'block_length', 'c_text', 'e_text'])
    """ note: block_length = 0-based 
    """

    def task2_align_fact(text, left, right, relations):

        """
        Implement fact alignment logic
        :param text:
        :param left:
        :param right:
        :param relations: Relations to consider for this text section.
        :return:
        TODO: refactor workflow to do text cleaning at the very last step (and remove from this function)
        TODO: fact alignment should be symetric - refactor code to align on sentence on the left as implemented on the right
        """
        try:
            left_text = text.split(left)[0]
            right_text = text.split(right)[-1]
            # ** Left align **
            # Ensure left part is not noise
            noise_pattern = ['(Reuters) -', 'MONTREAL', 'Analyst Ratings  This', '- - Sell Ratings',
                             'Analyst Recommendations  This ', '- - Net Margins']
            if not any([x in left_text for x in noise_pattern]):
                # Ensure there is no other fact on the left
                # when multiple causal relation are in the same text section
                extend_left = True
                if len(relations) > 1:
                    ref_index = text.index(left)
                    for r in relations:
                        clean_rel_text = clean_text(r.from_instance.text)
                        if left != clean_rel_text:
                            rel_index = text.find(clean_rel_text)
                            if (rel_index >= 0) and (rel_index < ref_index):
                                extend_left = False
                        clean_rel_text = clean_text(r.to_instance.text)
                        if left != clean_rel_text:
                            rel_index = text.find(clean_rel_text)
                            if (rel_index >= 0) and (rel_index < ref_index):
                                extend_left = False
                if extend_left:
                    left = left_text + left

            # **Right align**
            # Ensure there is no other fact on the right
            # when multiple causal relation are in the same text section
            extend_right = True
            if len(relations) > 1:
                ref_index = text.index(right)
                for r in relations:
                    clean_rel_text = clean_text(r.from_instance.text)
                    if right != clean_rel_text:
                        rel_index = text.find(clean_rel_text)
                        if (rel_index >= 0) and (rel_index > ref_index):
                            extend_right = False
                    clean_rel_text = clean_text(r.to_instance.text)
                    if left != clean_rel_text:
                        rel_index = text.find(clean_rel_text)
                        if (rel_index >= 0) and (rel_index > ref_index):
                            extend_right = False
            if extend_right:
                # If right fact is not a sentence, and there is still some text on the right
                if right[-1] != '.' and len(right_text) > 0:
                    # Look for the next sentence separator: '.' not followed by a digit (avoid 3.5%.)
                    # next_sentence = right_text.find('.')
                    next_sentence = re.search(r'\.(?!\d)', right_text)
                    if next_sentence is not None:
                        right += right_text[:next_sentence.span()[1]]

            return left, right
        except:
            pass

    def task2_print(index, task2_result_data, task_text, relations):
        """
        :param relations:
        :param index:
        :param task2_result_data:
        :param task_text:
        :return:
        """
        nonlocal writer2, fnp_task2_output_length
        t = task2_result_data
        # out_text = clean_text(t.text)
        out_text = clean_text(task_text)
        # keep track of sentence offsets
        sentence_offsets = [x for x, v in enumerate(out_text) if v == '\t']
        # get rid of sentence markers
        out_text = out_text.replace('\t', ' ')
        # Default case: use exact annotated segment for causal relationship
        out_cause = clean_text(t.cause)
        out_effect = clean_text(t.effect)
        # Use full sentence when causal relation span multiple lines
        # c_text and e_text are full sentence cause and effect gathered in task2_get_data
        if t.block_length > 0:
            out_cause = clean_text(t.c_text)
            out_effect = clean_text(t.e_text)

        # Retrieve cause & effect position and length in the text section
        c_start = out_text.find(out_cause)
        c_len = len(out_cause)
        e_start = out_text.find(out_effect)
        e_len = len(out_effect)
        if c_start == -1 or e_start == -1:
            tqdm.write(f'Error finding fact for {index} ({c_start},{e_start})')

        c_end = c_start + len(out_cause)
        e_end = e_start + len(out_effect)

        out_sentence = out_text.replace(out_cause, f'<e1>{out_cause}</e1>')
        out_sentence = out_sentence.replace(out_effect, f'<e2>{out_effect}</e2>')

        # HELPER: report spreadsheet: merge task 1 and task 2 output
        nonlocal relative_path, line_data, ld_index
        line = line_data[ld_index].index + 1
        # TODO add rel path to drive
        t2_url = f'https://drive{doc}?focus=sent~{line}'

        # HELPER: fact_alignment: cause & effect alignment rule on start & end of text section
        # Identify cases for cause & effect alignment rule on start & end of text section.
        out_sentence2 = out_sentence
        fa_right = out_text[0:min(c_start, e_start)].strip()
        fa_left = out_text[max(c_end, e_end):].strip()
        if len(fa_left) > 0 or len(fa_right) > 0:
            if c_start < e_start:
                try:
                    # Extend cause to left, effect to right
                    out_cause, out_effect = task2_align_fact(out_text, out_cause, out_effect, relations)
                except:
                    pass
            else:
                # Extend effect to left, cause to right
                out_effect, out_cause = task2_align_fact(out_text, out_effect, out_cause, relations)

            if out_cause and out_effect:
                c_start = out_text.find(out_cause)
                e_start = out_text.find(out_effect)
                c_end = c_start + len(out_cause)
                e_end = e_start + len(out_effect)

                new_fa_right = out_text[0:min(c_start, e_start)].strip()
                new_fa_left = out_text[max(c_end, e_end):].strip()

                out_sentence2 = out_text.replace(out_cause, f'<e1>{out_cause}</e1>')
                out_sentence2 = out_sentence2.replace(out_effect, f'<e2>{out_effect}</e2>')

                xlsx_file.add_fact_alignment(t2_url, index, out_text, out_sentence,
                                             fa_right, fa_left,
                                             out_sentence2,
                                             new_fa_right, new_fa_left)

        offset_sentence2 = sentence_offsets[0] if len(sentence_offsets) > 0 else ''
        offset_sentence3 = sentence_offsets[1] if len(sentence_offsets) > 1 else ''

        writer2.writerow([index, out_text, out_cause, out_effect,
                          offset_sentence2, offset_sentence3,
                          c_start, c_end, e_start, e_end,
                          out_sentence2])

        fnp_task2_output_length += 1

        # HELPER: report spreadsheet: merge task 1 and task 2 output
        xlsx_file.add_data_sets(t2_url, index, out_text, 1, out_cause, out_effect,
                                c_start, c_end, e_start, e_end, out_sentence2,
                                sentence_offsets=sentence_offsets)

    def task2_get_data(causal_relation: brat.BratAnnotationRelation, text_lines):
        """
        :param causal_relation:
        :param text_lines:
        :return: Task2ResultData = namedtuple('Task2ResultData',
                            ['cause', 'effect', 'start_line', 'block_length', 'c_text', 'e_text'])
                            c_text and e_text (cause text and effect text) are full sentences when cause and effect span
                            multiple lines
        """
        min_index = min(causal_relation.from_instance.lineno, causal_relation.to_instance.lineno)
        max_index = max(causal_relation.from_instance.lineno, causal_relation.to_instance.lineno)
        res_line = min_index
        res_len = max_index - min_index
        res_cause = causal_relation.from_instance.text
        res_effect = causal_relation.to_instance.text
        # Take full line for cause and effect when they span multiple lines in the text block
        res_c_text = text_lines[causal_relation.from_instance.lineno]
        res_e_text = text_lines[causal_relation.to_instance.lineno]
        return Task2ResultData(res_cause, res_effect, res_line, res_len, res_c_text, res_e_text)

    doc_index = 0
    with open(fnp_task1_output_pre, 'w', encoding='utf-8') as fp1, open(fnp_task2_output_pre, 'w',
                                                                        encoding='utf-8') as fp2:
        writer1 = csv.writer(fp1, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        writer2 = csv.writer(fp2, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        # CSV Headers
        # writer1.writerow(['URL', 'Index', 'Text', 'Gold'])
        # writer2.writerow(['URL', 'Index', 'Text', 'Cause', 'Effect'])
        writer1.writerow(['Index', 'Text', 'Gold'])
        writer2.writerow(['Index', 'Text', 'Cause', 'Effect',
                          'Offset_Sentence2', 'Offset_Sentence3',
                          'Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End',
                          'Sentence'])
        # CSV Rows
        for ann_file in tqdm(ann_files, desc='Processing', leave=True):
            if os.path.getsize(ann_file) == 0:
                continue
            ann_file_txt = ann_file.replace('.ann', '.txt')
            af = brat.BratAnnotatedFile(ann_file_txt)
            csv_filename = os.path.basename(ann_file_txt)

            relative_path = get_relative_path(ann_file_txt)

            if 'Discard' in af.annotation_types:
                continue
            else:
                doc_index += 1

            # CSV index reset for each document
            csv_index = 0

            # Create the set of lines to remove from the text (those marked with Remove annotations)
            # to eventually build the list of lines to process (note: line indexes are 0-based)
            lines_to_remove = set()
            if 'Remove' in af.annotation_types:
                lines_to_remove = {af.annotations[ann_key].lineno for ann_key in af.annotation_types['Remove'].keys()}
            lines = list(set(range(len(af.lines_text))) - lines_to_remove)
            # set() is not ordered >> order lines
            lines.sort()

            # Build the map of lines including their index (0-based), text and list of related causal relationships
            # Start by initializing the map with line index, text and empty list of causal relationship
            line_data = [LineData(i, af.lines_text[i], []) for i in lines]
            # Add causal relationships to their respective lines (when they exists)
            # First, get the list of relations with their respective block starting line
            # Note: ensure only BratAnnotationRelation are considered as the collection also include a 'Cause'
            # TextBoundAnnotation (annotation bug)
            if 'Cause' in af.annotation_types:
                causal_list = [CausalData(relation, min(relation.from_instance.lineno, relation.to_instance.lineno))
                               for relation in af.annotation_types['Cause'].values()
                               if isinstance(relation, brat.BratAnnotationRelation)]
                # Then add relations to their respective lines in the map
                for causal in causal_list:
                    causal_found = False
                    for ld in line_data:
                        if causal.start_index == ld.index:
                            causal_found = True
                            ld.add_causal_relation(causal.relation)
                    # Debug: ensure all relation are mapped in the text
                    if not causal_found:
                        tqdm.write(f'{af.txt_filename}: Relation not mapped to text: {causal.relation}')
            # else:
            #     causal_list = []

            # Process the text blocks to print for Task 1 & 2
            ld_index = 0
            ld_length = len(line_data)

            # Generate a sequence of block lengths using similar distribution as for causal blocks length
            # This is for building non causal blocks
            block_iter = iter(random.choices(block_lengths, target_weights, k=len(line_data) + 1))

            while ld_index < ld_length:
                ld = line_data[ld_index]
                csv_index += 1
                # Line has one or more causal relation (could be cause or effect or both)
                if ld.has_causal_relation():
                    # Bug fix - when a relation involves 2 or 3 lines (e.g. 1 >> 3), relations in lines 2 and/or 3 were ignored.
                    # If one of the causal block span accross multiple lines, ensure to also capture relations starting from those
                    # lines as they will be skipped eventually when moving to the next block to process
                    # (i.e. "ld_index += text_size + 1" later in this code)
                    # Note: by construction, relations returned by relations[].causal_relations() start from or end to the current
                    # line and their other part is later in the text (not earlier). i.e. relations are not oriented cause to effect
                    # in causal_relations()

                    # Get the text in-scope of this(these) relation(s) as well as any relations in the next lines within the limit of 3 lines.
                    # The resulting text section will be used for both Task 1 and Task 2 `Text` field.
                    # Collect all relations in that scope to generate the related text block.

                    def get_max_interval_size(relations):
                        """ Helper: return the max number of lines between source and target in a list of relations
                        Note: size is 0 when source and target on the same line.
                        """
                        line_intervals = [sorted([r.from_instance.lineno, r.to_instance.lineno]) for r in relations]
                        return max([b - a for a, b in line_intervals])

                    # Collect relations on this line
                    task_relations = ld.causal_relations()

                    text_size = get_max_interval_size(ld.causal_relations())
                    if text_size > 2:
                        tqdm.write(f'WARNING: a relation exceed the max length in {relative_path}:{ld.index + 1}')
                    # More than 1 line involved
                    # Consider relations on the second line and extend the text_size where appropriate
                    if text_size >= 1 and line_data[ld_index + 1].has_causal_relation():
                        task_relations.extend(line_data[ld_index + 1].causal_relations())
                        # Extend text_size
                        text_size_2 = get_max_interval_size(line_data[ld_index + 1].causal_relations())
                        if text_size == 1:
                            # add size of new relation
                            text_size += text_size_2
                        elif text_size_2 == 2:
                            # size is already 2 lines only add 1 line when new relation also span accross 2 lines.
                            text_size += 1
                        # Warning if text section goes over the limits
                        if text_size_2 > 1:
                            tqdm.write(
                                f'WARNING: a relation makes the text section larger than the limit in {relative_path}:{ld.index + 2}')

                    # 3 lines involved
                    # Consider relations on the 3rd line and raise a warning if text section goes over the limits
                    if text_size >= 2 and line_data[ld_index + 2].has_causal_relation():
                        task_relations.extend(line_data[ld_index + 2].causal_relations())
                        # Get the final text_size
                        text_size_2 = get_max_interval_size(line_data[ld_index + 2].causal_relations())
                        text_size += text_size_2
                        # Warning if text section goes over the limits
                        if text_size_2 > 0:
                            tqdm.write(
                                f'WARNING: a relation makes the text section larger than the limit in {relative_path}:{ld.index + 3}')

                    # Time to build the text section
                    task_text = line_data[ld_index].text
                    for i in range(1, text_size + 1):
                        task_text = '\t'.join([task_text, line_data[ld_index + i].text])

                    # Time to build all the text block for all the collected relations
                    causal_blocks = [task2_get_data(relation, af.lines_text) for relation in task_relations]

                    # Output task 1
                    task1_print(fnp2020_index(doc_index, csv_index), task_text, '1')

                    # Output task 2
                    # When multiple relations in a text section, generate sub index for task 2 so that
                    # task 1 and 2 share the same main index.
                    # Note: sub index generation is triggered if sub_index parameter != 0
                    for sub_index, block in enumerate(causal_blocks, 1 if len(causal_blocks) > 1 else 0):
                        task2_print(fnp2020_index(doc_index, csv_index, sub_index), block, task_text, task_relations)

                    # Move to next line to process
                    ld_index += text_size + 1
                else:
                    # No causal relations in the line
                    csv_text = ld.text
                    block_length = next(block_iter) - 1
                    real_block_length = 0
                    ld_index += 1
                    # Build multiple line block when randomly chosen and next lines have no causal or eof reached
                    while ld_index < ld_length and not line_data[ld_index].has_causal_relation() and block_length > 0:
                        csv_text = '\t'.join([csv_text, line_data[ld_index].text])
                        block_length -= 1
                        ld_index += 1
                        real_block_length += 1
                    # Output task 1
                    task1_print(fnp2020_index(doc_index, csv_index), csv_text, '0')
                    # Accumulate block length distribution for text without causal relationship
                    non_causal_lengths[real_block_length] += 1

    def field_separator_normalize(file_in: str, file_out: str, p_bar: tqdm):
        with open(file_in, 'r', encoding="utf-8") as fpi, open(file_out, "w", encoding="utf-8") as fpo:
            for line in fpi:
                fpo.write(line.replace('\t', '; '))
                p_bar.update()
        os.remove(file_in)

    # Post-processing - separator = "; " instead of "\t"
    with tqdm(desc='Normalizing', total=fnp_task1_output_length + fnp_task2_output_length, leave=True) as progress_bar:
        field_separator_normalize(fnp_task1_output_pre, fnp_task1_output, progress_bar)
        field_separator_normalize(fnp_task2_output_pre, fnp_task2_output, progress_bar)

    # print(causal_weights)
    try:
        print('Causal block length distribution')
        print(target_weights)
        total = sum(non_causal_lengths)
        print('Non Causal block length distribution')
        print([v / total for v in non_causal_lengths])

        print(f'Task 1: {fnp_task1_output_length}')
        print(f'Task 2: {fnp_task2_output_length}')

        xlsx_file.info_print("Processing details")
        xlsx_file.info_print('Causal block length distribution')
        xlsx_file.info_print(target_weights)
        xlsx_file.info_print('Non Causal block length distribution')
        xlsx_file.info_print([v / total for v in non_causal_lengths])
        xlsx_file.info_print(['Annotated Files', len(ann_files)])
        xlsx_file.info_print(['Task 1', fnp_task1_output_length,
                              fnp_task1_output_distrib[0] / fnp_task1_output_length,
                              fnp_task1_output_distrib[1] / fnp_task1_output_length])
        xlsx_file.info_print(['Task 2', fnp_task2_output_length])

        xlsx_file.close()

    except Exception as E:
        print(E)


if __name__ == '__main__':

    root = Path(__file__).parent
    parser = ArgumentParser()
    parser.add_argument("-i", type=str,  default='brat_files', help="input directory containing files to process")
    parser.add_argument("-o", type=str, default='tmp', help="output directory for the processed files")

    args = parser.parse_args()

    INPUT_DIR = args.i
    OUTPUT_DIR = args.o

    # get brat collections
    data_collections = [os.path.basename(x) for x in glob.glob(os.path.join(root, INPUT_DIR, "*")) if os.path.isdir(x)]

    # build the metadata list of filename to be processed
    docdict = {}
    for collection in data_collections:
        docdict[collection] = [os.path.basename(filename) for filename in glob.glob(os.path.join(root, INPUT_DIR, collection, '*.txt'))]
    print(docdict)

    for collection, docfile in docdict.items():
        os.makedirs(os.path.join(OUTPUT_DIR, collection), exist_ok=True)

        main('host', INPUT_DIR, collection, docfile, os.path.join(OUTPUT_DIR, collection))
