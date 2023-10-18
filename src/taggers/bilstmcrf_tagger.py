import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from taggers.base import TextTagger
from typing import List

from taggers.base import lookup_model

from flair.models import SequenceTagger
from loguru import logger

from models.document import Document
from methods.bilstmcrf import flair_utils
from tokenizer.base import Tokenizer

class BiLSTMCRFTagger(TextTagger):
    
    def __init__(self, model, tokenizer: Tokenizer, mini_batch_size=256, verbose=False):
        self.tokenizer = tokenizer
        self.mini_batch_size = mini_batch_size
        self.verbose = verbose

        model_file = lookup_model(model)
        logger.info('Load flair model from {}'.format(model_file))
        self.tagger = SequenceTagger.load(model_file)
        logger.info('Finish loading flair model.')

    def annotate(self, documents: List[Document]) -> List[Document]:
        flair_sents, parsed_docs = flair_utils.standoff_to_flair_sents(
            docs=documents,
            tokenizer=self.tokenizer,
            verbose=self.verbose
        )

        self.tagger.predict(flair_sents, mini_batch_size=self.mini_batch_size, verbose=self.verbose)

        annotated_docs = flair_utils.flair_sents_to_standoff(flair_sents, parsed_docs)
        return annotated_docs

    @property
    def tags(self):
        bio_tag_names = self.tagger.tag_dictionary.get_items()
        bio_tag_names.remove('<unk>')
        bio_tag_names.remove('<START>')
        bio_tag_names.remove('<STOP>')
        bio_tag_names.remove('O')

        tags = set()
        for bio_tag in bio_tag_names:
            name = bio_tag.split('-', maxsplit=1)[1]
            tags.add(name)
        return list(tags) 