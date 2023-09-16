import flair
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from flair.embeddings import (FlairEmbeddings, PooledFlairEmbeddings, StackedEmbeddings, TokenEmbeddings, WordEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from functools import partial
from loguru import logger
from methods.blistcrf.flair_utils import flair_sents_to_standoff, standoff_to_flair_sents, FilteredCorpus
from os.path import join
from utils.corpus_loader import CorpusLoader, BASE_PATH
from utils.train_utils import generate_model_folder_name, save_predictions
from tokenizer.base import TokenizerFactory
from typing import List

def _predict_ignored(sents):
    for sent in sents:
        for token in sent:
            token.add_tag('ner', 'O')

def make_predictions(tagger, filtered_corpus: FilteredCorpus):
    tagger.predict(filtered_corpus.train)
    tagger.predict(filtered_corpus.dev)
    tagger.predict(filtered_corpus.test)

    _predict_ignored(filtered_corpus.train_ignored)
    _predict_ignored(filtered_corpus.dev_ignored)
    _predict_ignored(filtered_corpus.test_ignored)

def get_embeddings(
    pooled: bool,
    contextual_forward_path: str = None,
    contextual_backward_path: str = None,
) -> List[TokenEmbeddings]:
    logger.info('Use Italian embeddings')
    word_embeddings = 'it'
    contextual_forward = 'news-forward'
    contextual_backward = 'news-backward'

    if contextual_forward_path:
        contextual_forward = contextual_forward_path
    if contextual_backward_path:
        contextual_backward = contextual_backward_path

    if pooled:
        logger.info('Use PooledFlairEmbeddings with mean pooling')
        ContextualEmbeddings = partial(PooledFlairEmbeddings, pooling='mean')
    else:
        logger.info('Use FlairEmbeddings')
        ContextualEmbeddings = FlairEmbeddings

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings(word_embeddings),
        ContextualEmbeddings(contextual_forward),
        ContextualEmbeddings(contextual_backward),
    ]

    return embedding_types

def get_model(
        corpus: flair.data.Corpus, 
        pooled_contextual_embeddings: bool,
        contextual_forward_path: str = None,
        contextual_backward_path: str = None,
    ):

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type = tag_type)

    embedding_types: List[TokenEmbeddings] = get_embeddings(
        pooled=pooled_contextual_embeddings,
        contextual_forward_path=contextual_forward_path,
        contextual_backward_path=contextual_backward_path
    )
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)
    return tagger

def run_blistcrf():
    logger.info('Starting...')

    # Load Corpus
    logger.info('Loading corpus...')
    corpus = CorpusLoader().load_corpus(BASE_PATH)
    logger.info('Loaded corpus: {}'.format(corpus))

    # Create Tokenizer
    tokenizer = TokenizerFactory()

    # Create Model Folder
    model_folder = generate_model_folder_name(corpus.name)
    os.makedirs(model_folder, exist_ok = True)

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

    # Train model
    logger.info('Training model...')

    tagger = get_model(
        flair_corpus,
        pooled_contextual_embeddings = False,
    )

    trainer = ModelTrainer(tagger, flair_corpus)

    trainer.train(
        base_path = join(model_folder),
        max_epochs = 3,
        mini_batch_size = 4,
        monitor_train = False,
        train_with_dev = False,
        embeddings_storage_mode = 'none',
    )

    if not False:
        # Model performance is judged by dev data, so we also pick the best performing model
        # according to the dev score to make our final predictions.
        tagger = SequenceTagger.load(join(model_folder, 'best-model.pt'))
    else:
        # Training is stopped if train loss converges - here, we do not have a "best model" and
        # use the final model to make predictions.
        pass

    logger.info('Make predictions...')
    make_predictions(tagger, flair_corpus)

    save_predictions(
        corpus_name = corpus.name,
        run_id = 'test',
        train = flair_sents_to_standoff(train_sentences, train_documents),
        dev = flair_sents_to_standoff(dev_sentences, dev_documents),
        test = flair_sents_to_standoff(test_sentences, test_documents)
    )



if __name__ == '__main__':
    run_blistcrf()