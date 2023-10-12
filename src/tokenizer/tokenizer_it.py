import os
import spacy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tokenizer.base import Tokenizer

NLP = spacy.load('it_core_news_sm')
class TokenizerIT(Tokenizer):

    def parse_text(self, text: str) -> spacy.tokens.doc.Doc:
        return NLP(text)