import os
import spacy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from abc import ABC, abstractmethod
from loguru import logger
from typing import Iterable

NLP = spacy.load('it_core_news_sm')

class Tokenizer(ABC):

    def __init__(self, disable: Iterable[str] = ()):
        """Tokenizer base class.

        Parameters
        ----------
        disable : Iterable[str]
            Steps of the spacy pipeline to disable.
            See: https://spacy.io/usage/processing-pipelines/#disabling

        """
        self.disable = disable

    def parse_text(self, text: str) -> spacy.tokens.doc.Doc:
        return NLP(text)


class TokenizerFactory():
    """Construct tokenizer instance per corpus. Currently, only the 'ons' corpus uses a custom
    spaCy tokenizer.

    For all other corpora, a wrapper around the default English spaCy tokenizer is used.
    """

    @staticmethod
    def tokenizer(disable: Iterable[str] = ()):
        from tokenizer.tokenizer_it import TokenizerIT
        return TokenizerIT(disable=disable)
    
    def parse_text(self, text: str) -> spacy.tokens.doc.Doc:
        return NLP(text)
