import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from functools import reduce
from typing import List, Tuple
from flair.data import Sentence, Token

from torch.nn.modules.module import _addindent

from models.document import Document
from utils.tagging_utils import ParsedDoc, sents_to_standoff, standoff_to_sents
from tokenizer.base import Tokenizer

def standoff_to_flair_sents(
    docs: List[Document],
    tokenizer: Tokenizer,
    verbose = False
) -> Tuple[List[Sentence], List[ParsedDoc]]:
    
    sents, parsed_docs = standoff_to_sents(docs = docs, tokenizer = tokenizer, verbose = verbose)

    flair_sents = []

    for sent in sents:
        flair_sent = Sentence()

        for token in sent:
            if token.text.isspace():
                # spaCy preserves consecutive whitespaces, while flair ignores them.
                # This would make a round-trip standoff -> token -> standoff impossible.
                # To accommodate whitespace tokens with flair, we add a special token.
                tok = Token('<SPACE>')

            else:
                tok = Token(token.text)

            tok.add_tag(tag_type='ner', tag_value=token.label)
            flair_sent.add_token(tok)
            
        flair_sents.append(flair_sent)

    return flair_sents, parsed_docs


def flair_sents_to_standoff(
    tagged_flair_sentences: List[Sentence],
    docs: List[ParsedDoc]
) -> List[Document]:

    sentence_tags = []
    
    for sent in tagged_flair_sentences:
        sentence_tags.append([
            # We introduced special space tokens in `standoff_to_flair_sents`.
            # If the model erreonously tagged those, we swap the label to O.
            token.get_tag('ner').value if token.text != '<SPACE>' else 'O' for token in sent
        ])

    return sents_to_standoff(sentence_tags, docs)