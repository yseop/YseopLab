# Cross-Lingual Few-Shot Relation Extraction for Pharmacovigilance in French, German, and Japanese

Yseop participated in
* Task 2a for French and Japanese
* Task 2b for Japanese only


## Task 2a Japanese NER
### Training Data

* `data-check.ipynb`: check for annotation errors
* `tokenize.ipynb`: error introduced by the tokenizer
* `baseline.ipynb`: rule based baseline
* `task2a-fr.ipynb`: solution for task 2a french
* `finetune-train-split.ipynb`: ner model fine tuned on 80% for the train data. https://huggingface.co/yseop/SMM4H2024_Task2a_ja
* `finetune-train-dev.ipynb`: ner model not submitted for the shared task
* `task2b-ja.ipynb`: classification model for task 2b japanese. https://huggingface.co/yseop/SMM4H2024_Task2b_ja
* `task2-ja-submission.ipynb`: relation extraction (classification model) depends on the output of ner model
