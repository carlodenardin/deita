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

@app.get("/api/python")
def hello_world():
    text = (
        "Il paziente Mario Rossi ha 22 anni e vive a Milano."
    )
    documents = [
        Document(name='doc_01', text=text)
    ]
    model = 'src/best-model.pt'

    tokenizer = TokenizerFactory().tokenizer(corpus='ehr')

    print("AAA")
    tagger = BlistCRFTagger(model = model, tokenizer = tokenizer, verbose = False)

    annotated_docs = tagger.annotate(documents)
    return {"output": annotated_docs[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)