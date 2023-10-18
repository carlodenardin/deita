from flair.embeddings import (FlairEmbeddings, PooledFlairEmbeddings, StackedEmbeddings, TokenEmbeddings, WordEmbeddings)
from functools import partial
from loguru import logger
from typing import List

def get_embeddings(
    language: str = 'it',
    pooled: bool = False,
    contextual_forward_path: str = None,
    contextual_backward_path: str = None,
):

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

    embeddings: List[TokenEmbeddings] = [
        WordEmbeddings(language),
        ContextualEmbeddings(contextual_forward),
        ContextualEmbeddings(contextual_backward),
    ]

    return embeddings
