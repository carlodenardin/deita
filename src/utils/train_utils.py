import numpy as np
import os
import scipy.stats
import sys
import utils.brat as brat
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

from models.document import Document
from os.path import dirname, join
from typing import List

OUTPUT_PATH = join(dirname(__file__), '../../output')

def generate_model_folder_name(corpus_name: str, run_id: str) -> str:
    return join(OUTPUT_PATH, corpus_name, run_id)

def _save_predictions(path, documents: List[Document]):
    os.makedirs(path, exist_ok=True)

    for doc in documents:
        brat.write_brat_annotations(doc.annotations, join(path, '{}.ann'.format(doc.name)))

def save_predictions(
    corpus_name,
    run_id,
    train: List[Document] = None,
    test: List[Document] = None,
    dev: List[Document] = None
):

    for part_name, part_docs in zip(['train', 'test', 'dev'], [train, test, dev]):
        if not part_docs:
            continue

        logger.info('Write {}.{}.{} predictions (N = {})'.format(corpus_name, run_id, part_name, len(part_docs)))
        base_path = generate_model_folder_name(corpus_name, run_id)
        _save_predictions(join(base_path, part_name), part_docs)