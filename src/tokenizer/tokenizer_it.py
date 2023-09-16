import spacy

NLP = spacy.load("it_core_news_sm")

class TokenizerIT():

    def parse_text(self, text: str) -> spacy.tokens.doc.Doc:
        return NLP(text)