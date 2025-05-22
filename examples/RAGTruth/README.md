# TODO

RAGTruth

3 different tasks:
Summary
Data 2 text
QA

Composed by a total of 2965 sources or original documents/ tasks.

- 943 Summary
- 1033 Data 2 text
- 989 QA

For very one of this task RAGTruth has created an answer given by an LLM, they have used 6 types of LLMs.
Thus the DB becomes 6 times bigger.

Total of 17790 documents.

- 5658 Summary
- 6198 Data 2 text
- 5934 QA

All of this can be found in the dataset folder.
Furthermore the dataset is divided into 2 splits, train and test.

The test split is composed by a total of 2700 LLM asnwers.
900 from summary
900 from data 2 text
900 from QA

All the rest of the documents are in the train split.

Following the instruction in RAGTruth and their github we generated 10 samples for every LLM answer contained in the test set. To be able to test our method, SELFCHECKGPT and BERTSCore accordingly to our paper.

All run locally, the samples script has been given, The cost is 0

LLM-as-a-Judge cost to establish
