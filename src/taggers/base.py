import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from abc import ABC, abstractmethod
from loguru import logger
from models.document import Document
from typing import List

class TextTagger(ABC):

    @abstractmethod
    def annotate(self, documents: List[Document]) -> List[Document]:
        pass

    @property
    @abstractmethod
    def tags(self) -> List[str]:
        pass