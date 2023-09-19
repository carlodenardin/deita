import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/'))

from flair.models import SequenceTagger
from loguru import logger
from methods.blistcrf.flair_utils import flair_sents_to_standoff, standoff_to_flair_sents, FilteredCorpus
from utils.corpus_loader import CorpusLoader, BASE_PATH
from utils.train_utils import save_predictions
from tokenizer.base import TokenizerFactory
from models.document import Document

def run_predictions_blistcrf():
    logger.info('Starting...')

    # Load Corpus
    logger.info('Loading corpus...')
    corpus = CorpusLoader().load_corpus(BASE_PATH)
    logger.info('Loaded corpus: {}'.format(corpus))

    # Create Tokenizer
    tokenizer = TokenizerFactory()

    # Get Sentences
    logger.info('Getting sentences...')
    train_sentences, train_documents = standoff_to_flair_sents(corpus.train, tokenizer)
    test_sentences, test_documents = standoff_to_flair_sents(corpus.test, tokenizer)
    dev_sentences, dev_documents = standoff_to_flair_sents(corpus.dev, tokenizer)
    logger.info('Train sentences: {}, Train docs: {}'.format(len(train_sentences), len(train_documents)))
    logger.info('Test sentences: {}, Test docs: {}'.format(len(test_sentences), len(test_documents)))
    logger.info('Dev sentences: {}, Dev docs: {}'.format(len(dev_sentences), len(dev_documents)))

    # Create flair corpus
    flair_corpus = FilteredCorpus(train = train_sentences, dev = dev_sentences, test = test_sentences)
    logger.info(flair_corpus)

    # Predict
    tagger = SequenceTagger.load('final-model.pt')

    logger.info('Make predictions...')
    
    tagger.predict(flair_corpus.train)
    tagger.predict(flair_corpus.dev)
    tagger.predict(flair_corpus.test)

    save_predictions(
        corpus_name = corpus.name,
        run_id = 'test',
        train = flair_sents_to_standoff(train_sentences, train_documents),
        dev = flair_sents_to_standoff(dev_sentences, dev_documents),
        test = flair_sents_to_standoff(test_sentences, test_documents)
    )

def run_predictions_crf():

    from taggers.crf_tagger import CRFTagger

    # Create some text
    text = (
        "Il paziente Carlo è residente in Via P. Soccol 20 A, 32021, Agordo (BL) nato il 12/18/1997 a Feltre. Presenta diversi sintomi, ora si trova nella stanza 120."
    )

    # Wrap text in document
    documents = [
        Document(name='doc_01', text=text)
    ]

    # Select downloaded model
    model = 'model.pickle'

    # Instantiate tokenizer
    tokenizer = TokenizerFactory().tokenizer(disable=("tagger", "ner"))

    # Load tagger with a downloaded model file and tokenizer
    tagger = CRFTagger(model, tokenizer)

    # Annotate your documents
    annotated_docs = tagger.annotate(documents)

    from pprint import pprint

    first_doc = annotated_docs[0]
    pprint(first_doc.annotations)

    from utils.replace_phi import mask_annotations

    masked_doc = mask_annotations(first_doc)
    print(masked_doc.text)

if __name__ == '__main__':
    #run_predictions_blistcrf()
    run_predictions_crf()