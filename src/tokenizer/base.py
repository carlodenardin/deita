from abc import ABC, abstractmethod
from typing import Iterable

import spacy
from loguru import logger

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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

    @abstractmethod
    def parse_text(self, text: str) -> spacy.tokens.doc.Doc:
        pass


class TokenizerFactory():
    """Construct tokenizer instance per corpus. Currently, only the 'ons' corpus uses a custom
    spaCy tokenizer.

    For all other corpora, a wrapper around the default English spaCy tokenizer is used.
    """

    @staticmethod
    def tokenizer(corpus: str, disable: Iterable[str] = ()):
        from tokenizer.tokenizer_it import TokenizerIT
        return TokenizerIT(disable=disable)