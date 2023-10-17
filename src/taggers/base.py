import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abc import ABC, abstractmethod
from os.path import isfile
from pathlib import Path
from typing import List


from abc import ABC, abstractmethod
from loguru import logger
from models.document import Document
from typing import List

def lookup_model(model):
    """Get model file from model name. If `model` is a path to an existing file on the file system,
    `model` is returned instead. Otherwise, this function searches for a model in the model download
    cache.
    """
    if isfile(model):
        return model

    return model

class TextTagger(ABC):

    @abstractmethod
    def annotate(self, documents: List[Document]) -> List[Document]:
        pass

    @property
    @abstractmethod
    def tags(self) -> List[str]:
        pass