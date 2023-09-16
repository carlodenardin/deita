from collections import namedtuple

class Annotation(namedtuple('Annotation', ['text', 'start', 'end', 'tag', 'doc_id', 'ann_id'])):

    __slots__ = ()

    def __new__(cls, text, start, end, tag, doc_id = '', ann_id = ''):
        return super(Annotation, cls).__new__(cls, text, start, end, tag, doc_id, ann_id)

    def __str__(self) -> str:
        return f'Annotation(text={self.text}, start={self.start}, end={self.end}, tag={self.tag}, doc_id={self.doc_id}, ann_id={self.ann_id})'
    
    def __unicode__(self) -> str:
        return self.__str__()
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        return self.text == other.text and self.start == other.start and self.end == other.end and self.tag == other.tag and self.doc_id == other.doc_id and self.ann_id == other.ann_id
    
    def __hash__(self) -> int:
        return hash((self.text, self.start, self.end, self.tag, self.doc_id, self.ann_id))