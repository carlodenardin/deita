import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.annotation import Annotation
from typing import List


class Document:

    def __init__(self, name: str, text: str, annotations: List[Annotation] = ()):
        self.name = name
        self.text = text
        self.annotations = annotations

    def __str__(self) -> str:
        return f"Document(name = {self.name}): Chars: {len(self.text)}, Annotations: {len(self.annotations)}"
    
    def __unicode__(self) -> str:
        return self.__str__()
    
    def __repr__(self) -> str:
        return self.__str__()