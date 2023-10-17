import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from models.document import Document
from taggers.blistcrf_tagger import BlistCRFTagger
from tokenizer.base import TokenizerFactory

class Input(BaseModel):
    text: str

app = FastAPI()

@app.post("/api/python/v1")
def hello_world(input: Input):
    documents = [
        Document(name='doc_01', text=input.text)
    ]
    model = 'src/bilstmcrf.pt'

    tokenizer = TokenizerFactory().tokenizer(corpus='ehr')

    tagger = BlistCRFTagger(model = model, tokenizer = tokenizer, verbose = False)

    annotated_docs = tagger.annotate(documents)

    return {"output": annotated_docs[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)