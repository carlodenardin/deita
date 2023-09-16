import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.document import Document
from typing import List

class Corpus:

    def __init__(self, train: List[Document], test: List[Document], dev: List[Document], name):
        self.train = train
        self.test = test
        self.dev = dev
        self.name = name
    
    def __str__(self) -> str:
        return f"Corpus(name = {self.name}): Number of Documents({len(self.train)}/{len(self.test)}/{len(self.dev)})"
    
    def __unicode__(self) -> str:
        return self.__str__()
    
    def __repr__(self) -> str:
        return self.__str__()