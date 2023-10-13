import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from loguru import logger
from tokenizer.base import TokenizerFactory
from utils.corpus_loader import CorpusLoader, BASE_PATH
from utils.tagging_utils import standoff_to_sents
from methods.crf import crf_labeler, crf_utils
from utils import tagging_utils
from utils.train_utils import generate_model_folder_name, save_predictions
import sklearn_crfsuite


def run_crf(iteration):
    corpus = CorpusLoader().load_corpus(BASE_PATH)
    tokenizer = TokenizerFactory().tokenizer('ehr')
    logger.info('Loaded corpus: {}'.format(corpus))

    # Select randmly only a subset of the corpus for training '''len(corpus.train) * (iteration * 20) // 100'''
    corpus.train = random.sample(corpus.train, len(corpus.train) * (iteration * 10) // 100)

    # Create Model Folder
    model_folder = generate_model_folder_name(corpus.name, f'crf_{(iteration * 10)}')
    os.makedirs(model_folder, exist_ok = True)

    logger.info('Get and Tokenize sentences...')
    train_sents, train_docs = standoff_to_sents(corpus.train, tokenizer, verbose = True)
    dev_sents, dev_docs = tagging_utils.standoff_to_sents(corpus.dev, tokenizer, verbose = True)
    test_sents, test_docs = standoff_to_sents(corpus.test, tokenizer, verbose = True)

    logger.info('Compute features...')
    feature_extractor = crf_labeler.liu_feature_extractor
    X_train, y_train = crf_labeler.sents_to_features_and_labels(train_sents, feature_extractor)  
    X_dev, y_dev = crf_labeler.sents_to_features_and_labels(dev_sents, feature_extractor)
    X_test, _ = crf_labeler.sents_to_features_and_labels(test_sents, feature_extractor)

    logger.info(f'len(X_train) = {len(X_train)}')
    logger.info(f'len(y_train) = {len(y_train)}')
    logger.info(f'len(X_dev) = {len(X_dev)}')
    logger.info(f'len(X_test) = {len(X_test)}')

    crf = sklearn_crfsuite.CRF(
        algorithm = 'lbfgs',
        c1 = 0.1,
        c2 = 0.1,
        max_iterations = 100,
        all_possible_transitions = True
    )

    logger.info('Start training... {}'.format(crf))
    crf.fit(X_train, y_train)

    logger.info('Make predictions...')
    y_pred_train = crf.predict(X_train)
    y_pred_dev = crf.predict(X_dev)
    y_pred_test = crf.predict(X_test)

    save_predictions(
        corpus_name = corpus.name,
        run_id = f'crf_{(iteration * 10)}/predictions',
        train = tagging_utils.sents_to_standoff(y_pred_train, train_docs),
        dev = tagging_utils.sents_to_standoff(y_pred_dev, dev_docs),
        test = tagging_utils.sents_to_standoff(y_pred_test, test_docs)
    )

    logger.info('Save model artifacts...')
    labels = list(crf.classes_)
    labels.remove('O')
    crf_utils.save_bio_report(y_dev, y_pred_dev, labels, model_folder)
    crf_utils.save_transition_features(crf, model_folder)
    crf_utils.save_state_features(crf, model_folder)
    crf_utils.persist_model(crf, model_folder)

if __name__ == '__main__':
    for i in range(1, 11):
        run_crf(i)