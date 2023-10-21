import warnings
import os
import sys
import csv
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union
from sklearn import metrics


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Get a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()
   
    def get_dev_examples(self, data_dir):
        """Get a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Get a collection of [`InputExample`] for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Get the list of labels for this data set"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Read a tab separated value file."""
        with open(input_file, 'r', encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter = "\t", quotechar=quotechar))



@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. 
                For single sequence tasks(e.g.: Text-Steganalysis), 
                only this sequence must be specified
        text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be 
                specified for train and dev examples, but not for text examples.
    """
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"



class TStega_processor(DataProcessor):
    """Processor for the Text Steganalysis data set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        # text_index = 1 if set_type == "test" else 0
        text_index = 0
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            # label = None if set_type == "test" else line[1]
            label = line[1]
            examples.append(InputExample(
                    guid=guid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label,
                )
            )
        return examples




def compute_metrics(preds, labels, phase):
    if phase == "dev":
        accuracy = metrics.accuracy_score(labels, preds)
        return {"accuracy": accuracy}
    elif phase == "test":
        accuracy = metrics.accuracy_score(labels, preds)
        precious = metrics.precision_score(labels, preds)
        recall = metrics.recall_score(labels, preds)
        F1_score = metrics.f1_score(labels, preds, average="weighted")
        return {
            "accuracy": accuracy, 
            "precious": precious,
            "-recall-": recall,
            "F1_score": F1_score,
        }
        
