import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
from typing import List

from loguru import logger

from models.document import Document
from utils import tagging_utils
import methods.crf.crf_labeler as crf_labeler
import methods.crf.crf_utils as crf_utils
from taggers.base import TextTagger
from tokenizer.base import Tokenizer


class CRFTagger(TextTagger):

    def __init__(self, model, tokenizer: Tokenizer, verbose=False):
        self.tokenizer = tokenizer
        self.feature_extractor = crf_labeler.liu_feature_extractor
        self.verbose = verbose

        model_file = model
        logger.info('Load sklearn-crfsuite model from {}'.format(model_file))
        with open(model_file, 'rb') as clf_file:
            self.tagger = pickle.load(clf_file)
        logger.info('Finish loading crf model.')

    def annotate(self, documents: List[Document]) -> List[Document]:
        sents, parsed_docs = tagging_utils.standoff_to_sents(
            docs=documents,
            tokenizer=self.tokenizer,
            verbose=self.verbose
        )

        X_features, _ = crf_labeler.sents_to_features_and_labels(sents, self.feature_extractor)
        y_pred = self.tagger.predict(X_features)
        annotated_docs = tagging_utils.sents_to_standoff(y_pred, parsed_docs)
        return annotated_docs

    @property
    def tags(self):
        bio_tag_names = self.tagger.classes_
        bio_tag_names.remove('O')

        targets = set()
        for target in bio_tag_names:
            name = target.split('-', maxsplit=1)[1]
            targets.add(name)

        return list(targets)