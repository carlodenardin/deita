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


def run_crf():
    corpus = CorpusLoader().load_corpus(BASE_PATH)
    tokenizer = TokenizerFactory().tokenizer('ehr')
    logger.info('Loaded corpus: {}'.format(corpus))

    # Create Model Folder
    model_folder = generate_model_folder_name(corpus.name, 'crf')
    os.makedirs(model_folder, exist_ok = True)

    logger.info('Get sentences...')
    train_sents, train_docs = standoff_to_sents(corpus.train, tokenizer, verbose=True)
    test_sents, test_docs = standoff_to_sents(corpus.test, tokenizer, verbose=True)

    logger.info('Compute features...')
    feature_extractor = crf_labeler.liu_feature_extractor
    X_train, y_train = crf_labeler.sents_to_features_and_labels(train_sents, feature_extractor)
    X_test, _ = crf_labeler.sents_to_features_and_labels(test_sents, feature_extractor)

    logger.info('len(X_train) = {}'.format(len(X_train)))
    logger.info('len(y_train) = {}'.format(len(y_train)))
    logger.info('len(X_test) = {}'.format(len(X_test)))

    crf = crf_labeler.SentenceFilterCRF(
        ignored_label='O',
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    logger.info('Start training... {}'.format(crf))
    crf.fit(X_train, y_train)

    logger.info('CRF classes: {}'.format(crf.classes_))

    logger.info('Make predictions...')
    y_pred_train = crf.predict(X_train)
    y_pred_test = crf.predict(X_test)

    save_predictions(corpus_name=corpus.name, run_id='crf',
                                train=tagging_utils.sents_to_standoff(y_pred_train, train_docs),
                                test=tagging_utils.sents_to_standoff(y_pred_test, test_docs))

    logger.info('Save model artifacts...')
    labels = list(crf.classes_)
    labels.remove('O')
    crf_utils.save_transition_features(crf, model_folder)
    crf_utils.save_state_features(crf, model_folder)
    crf_utils.persist_model(crf, model_folder)


if __name__ == '__main__':
    run_crf()