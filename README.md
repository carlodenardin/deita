# DeITA

This repository contains a custom implementation of the algorithms described in the repository [deidentify](https://github.com/nedap/deidentify).

The purpose of this repository is to provide methods that can be used to deidentify EHR data in a language that is different from English. The current implementation is for the Italian language.

The data folder contains the data used to train the model in particular the EHR corpus (generated with openapi and the faker library). In the repository [deidentify](https://github.com/nedap/deidentify) there are the documentations on how to generate the corpus in the brat-standoff format.

Run BiLSTM-CRF model training:
`python src/methods/bilstmcrf.py` 

Run CRF model training:
`python src/methods/crf.py`

Evaluate the model:
`python src/evaluation/evaluate_run.py it data/corpus/ehr/test data/corpus/ehr/test output/bilstmcrf_100/predictions/test`