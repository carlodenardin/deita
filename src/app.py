import os
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from models.document import Document
from taggers.blistcrf_tagger import BlistCRFTagger
from tokenizer.base import TokenizerFactory

from fastapi.middleware.cors import CORSMiddleware

class Input(BaseModel):
    text: str

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/python/v1")
def hello_world(input: Input):

    print('Input: ', input.text)

    documents = [
        Document(name='doc_01', text=input.text)
    ]

    model = 'src/bilstmcrf.pt'

    if os.path.isfile(model):
        print('Model exists')
    else:
        print('Model does not exist')
        return {"output": "Model does not exist"}

    print('Tokenizing...')
    tokenizer = TokenizerFactory().tokenizer(corpus='ehr')

    print('Tagging...')
    tagger = BlistCRFTagger(model = model, tokenizer = tokenizer, mini_batch_size = 32, verbose = False)

    print('Annotating...')
    annotated_docs = tagger.annotate(documents)

    return {"output": annotated_docs[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)