import flair
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from flair.embeddings import StackedEmbeddings, TokenEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from loguru import logger
from methods.blistcrf.flair_utils import flair_sents_to_standoff, standoff_to_flair_sents
from methods.blistcrf.embeddings import get_embeddings
from os.path import join
from utils.corpus_loader import CorpusLoader, BASE_PATH
from utils.train_utils import generate_model_folder_name, save_predictions
from tokenizer.base import TokenizerFactory
from typing import List
from flair.data import Corpus

def get_model(
        corpus: flair.data.Corpus, 
        language: str = 'it',
        pooled_contextual_embeddings: bool = False,
        contextual_forward_path: str = None,
        contextual_backward_path: str = None,
    ):

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type = tag_type)

    embedding_types: List[TokenEmbeddings] = get_embeddings(
        language = language,
        pooled = pooled_contextual_embeddings,
        contextual_forward_path = contextual_forward_path,
        contextual_backward_path = contextual_backward_path
    )
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size = 256,
        embeddings = embeddings,
        tag_dictionary = tag_dictionary,
        tag_type = tag_type
    )

    return tagger

def run_blistcrf(iteration):
    # Load Corpus
    logger.info('Loading corpus...')
    corpus = CorpusLoader().load_corpus(BASE_PATH)
    logger.info('Loaded corpus: {}'.format(corpus))

    # Create Tokenizer
    tokenizer = TokenizerFactory().tokenizer(corpus)

    # Select randmly only a subset of the corpus for training
    corpus.train = random.sample(corpus.train, len(corpus.train) * (iteration * 10) // 100)

    # Create Model Folder
    model_folder = generate_model_folder_name(corpus.name, f'bilstmcrf_{len(corpus.train)}')
    os.makedirs(model_folder, exist_ok = True)

    # Get Sentences
    logger.info('Get and Tokenize sentences...')
    train_sentences, train_documents = standoff_to_flair_sents(corpus.train, tokenizer)
    test_sentences, test_documents = standoff_to_flair_sents(corpus.test, tokenizer)
    dev_sentences, dev_documents = standoff_to_flair_sents(corpus.dev, tokenizer)
    logger.info('Train sentences: {}, Train docs: {}'.format(len(train_sentences), len(train_documents)))
    logger.info('Test sentences: {}, Test docs: {}'.format(len(test_sentences), len(test_documents)))
    logger.info('Dev sentences: {}, Dev docs: {}'.format(len(dev_sentences), len(dev_documents)))

    # Save train_sentences
    with open(join(model_folder, 'train_sentences.txt'), 'w') as f:
        f.write('\n'.join([str(s) for s in train_sentences]))
    # Create flair corpus
    flair_corpus = Corpus(train = train_sentences, dev = dev_sentences, test = test_sentences)
    logger.info(flair_corpus)

    # Save flair corpus   
    with open(join(model_folder, 'flair_train.txt'), 'w') as f:
        f.write('\n'.join([str(s) for s in flair_corpus.train]))

    # Train model
    logger.info('Training model...')

    tagger = get_model(
        corpus = flair_corpus,
        language = 'it',
        pooled_contextual_embeddings = False,
        contextual_forward_path = 'it-forward',
        contextual_backward_path = 'it-backward'
    )

    trainer = ModelTrainer(tagger, flair_corpus)

    trainer.train(
        base_path = join(model_folder),
        max_epochs = 1,
        mini_batch_size = 8,
        monitor_train = False,
        train_with_dev = False,
        embeddings_storage_mode = 'none',
    )

    tagger = SequenceTagger.load(join(model_folder, 'best-model.pt'))

    logger.info('Make predictions...')
    tagger.predict(flair_corpus.train)
    tagger.predict(flair_corpus.dev)
    tagger.predict(flair_corpus.test)

    save_predictions(
        corpus_name = corpus.name,
        run_id = f'bilstmcrf{(iteration * 20)}/predictions',
        train = flair_sents_to_standoff(train_sentences, train_documents),
        dev = flair_sents_to_standoff(dev_sentences, dev_documents),
        test = flair_sents_to_standoff(test_sentences, test_documents)
    )

if __name__ == '__main__':
    for i in range(1, 2):
        run_blistcrf(i)