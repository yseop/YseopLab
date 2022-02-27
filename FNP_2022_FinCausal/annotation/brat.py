""" FNP2020 BRAT reader
"""

import bisect
import logging

# Brat Annotations: ID conventions
# Reference https://brat.nlplab.org/standoff.html
# All annotations IDs consist of a single upper-case character identifying the annotation type and a number.
# The initial ID characters relate to annotation types as follows:
# T: text-bound annotation
# R: relation
# E: event
# A: attribute
# M: modification (alias for attribute, for backward compatibility)
# N: normalization [new in v1.3]
# #: note

# TODO:Refactoring real class hierarchy for Annotations entities
class BratAnnotationBase():
    def __init__(self, line):
        super().__init__()


class BratAnnotationNull():
    def __init__(self, line):
        super().__init__()
        self.id, self.type = line.split('\t')[0:2]
        self.line = self.type.split()[1:]
        self.type = self.type.split()[0]

    def __repr__(self):
        # return super().__repr__()(self):
        return f'{self.id} {self.type} {self.line}'


class BratAnnotationTextBound():
    def __init__(self, line):
        """
        continuous: T1	Organization 0 4	Sony
        discontinuous: T1	Location 0 5;16 23	North America
        """
        super().__init__()
        self.id, self.type, self.text = line.split('\t')
        # TODO:implement discontinuous text bound
        # keep the first part of the discontinuous text bound for now
        self.type = self.type.split(';')[0]
        self.type, self.start, self.end = self.type.split()
        self.lineno = -1

    def __repr__(self):
        # return super().__repr__()(self):
        return f'{self.id} {self.type} {self.start} {self.end} {self.lineno}'


class BratAnnotationRelation():
    def __init__(self, line):
        """
        R1  Cause Cause:T1 Effect:T2
        """
        super().__init__()
        # The ID is separated by a TAB character, and the relation type and arguments by SPACE.
        self.id, self.type = line.split('\t')[0:2]
        self.type, self.from_type, self.to_type = self.type.split()
        self.from_type, self.from_id = self.from_type.split(':')
        self.to_type, self.to_id = self.to_type.split(':')
        self.from_instance = None
        self.to_instance = None

    def __repr__(self):
        # return super().__repr__()(self):
        return f'{self.id} {self.type} {self.from_type}:{self.from_id} {self.to_type}:{self.to_id} [{self.from_instance} >> {self.to_instance}]'


class BratAnnotationNote():
    def __init__(self, line):
        """
        #1	AnnotatorNotes T9	To be reviewed by a financial analyst.
        """
        super().__init__()
        self.id, self.type, self.text = line.split('\t')
        self.type, self.noteid = self.type.split()

    def __repr__(self):
        # return super().__repr__()(self):
        return f'{self.id} {self.type} {self.noteid} {self.text}'


class BratAnnotatedFile:
    ANNOTATIONS_IDS = {
        # T: text-bound annotation
        "T": BratAnnotationTextBound,
        # R: relation
        "R": BratAnnotationRelation,
        # E: event
        "E": BratAnnotationNull,
        # A: attribute
        "A": BratAnnotationNull,
        # M: modification (alias for attribute, for backward compatibility)
        "M": BratAnnotationNull,
        # N: normalization [new in v1.3]
        "N": BratAnnotationNull,
        # #: note
        "#": BratAnnotationNote
    }

    def __init__(self, filename):
        super().__init__()

        self.txt_filename = filename
        self.ann_filename = filename.replace('.txt', '.ann')

        self.annotations = {}
        self.annotation_types = {}

        self._load_annotation()
        self._load_text()

        self._resolve_annotation_lines()
        self._resolve_relations()

    def _load_text(self):
        """ Load text content in a list of lines and keep track of each end of line offsets
        - self.lines_text is the list of lines
        - self.lines_index is the list of end of line offset (compared to file size)
        This is to eventually map annotations offsets with their respective line index
        """
        self.lines_text = []
        self.lines_index = []
        curr_index = 0
        with open(self.txt_filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                curr_index += len(line)
                self.lines_index.append(curr_index)
                self.lines_text.append(line.strip('\n'))

    def _load_annotation(self):
        """ Load related Brat annotations file to internal structures
        - self.annotations is the dict of annotation instances (i.e. key = annotation id,
        value = annotation object)
        Example: annotations['T1'] = <T1 Discard 0 3>
        - self.annotation_types is the dict of type of annotations (i.e. key = type, value = dico of annotation with key = id)
        Example: annotations_type['Discard'] = { 'T1': <T1 Discard 0 3>  }
        """
        with open(self.ann_filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                # Strip any leading space
                line = line.lstrip()
                # Strip any trailing newlines
                line = line.rstrip('\n')
                # Ensure annotation id is valid
                # (first character is the key in the ANNOTATIONS_IDS dict)
                if line[0] in self.ANNOTATIONS_IDS:
                    # Add annotation record to the annotation IDs dico
                    # Get the Annotation class using the key
                    # and initialize it with the line to parse
                    try:
                        new_annotation = self.ANNOTATIONS_IDS[line[0]](line)
                        self.annotations[new_annotation.id] = new_annotation
                        # Add annotation record to the annotation type dico
                        # First ensure dico is created for the given type
                        if not new_annotation.type in self.annotation_types:
                            self.annotation_types[new_annotation.type] = {}
                        self.annotation_types[new_annotation.type][new_annotation.id] = new_annotation
                    except:
                        logging.warning(f'Annotation parsing error: {self.ann_filename} {line}')
                        # os.sys.exit()

    def _resolve_annotation_lines(self):
        """ Iterate all annotation and resolve text bound annotation to their respective line numbers
        """
        for annotation in self.annotations.values():
            if isinstance(annotation, BratAnnotationTextBound):
                annotation.lineno = bisect.bisect_left(self.lines_index, int(annotation.start) + 1)

    def _resolve_relations(self):
        """ Iterate all annotation and resolve text bound annotations instance in relation annotations
        """
        for annotation in self.annotations.values():
            if isinstance(annotation, BratAnnotationRelation):
                annotation.from_instance = self.annotations[annotation.from_id]
                annotation.to_instance = self.annotations[annotation.to_id]