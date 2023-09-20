import glob
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.corpus import Corpus
from models.document import Document
from os.path import basename, dirname, join, normpath, splitext
from typing import List
import utils.brat as brat

BASE_PATH = join(dirname(__file__), '../../data/corpus/ehr')

def _get_basename(full_path):
        return splitext(basename(full_path))[0]

class CorpusLoader():
    
    @staticmethod
    def _load_folder(path) -> List[Document]:
        files = glob.glob(join(path, '*.ann'))
        files = sorted(files)

        documents = []
        for file in files:
            doc_name = _get_basename(file)
            annotations, text = brat.load_brat_document(path, doc_name)
            doc = Document(name = doc_name, text = text, annotations = annotations)
            documents.append(doc)

        return documents
    
    def load_corpus(self, path) -> Corpus:
        corpus_name = basename(path)

        train = self._load_folder(join(path, 'train'))
        test = self._load_folder(join(path, 'test'))
        dev = self._load_folder(join(path, 'dev'))

        return Corpus(train = train, test = test, dev = dev, name = corpus_name)
