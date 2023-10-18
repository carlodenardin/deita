import os
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from models.document import Document
from taggers.bilstmcrf_tagger import BiLSTMCRFTagger
from taggers.crf_tagger import CRFTagger
from tokenizer.base import TokenizerFactory

from fastapi.middleware.cors import CORSMiddleware

class Input(BaseModel):
    text: str

app = FastAPI()

origins = [
    "https://deita.vercel.app",
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

    document = [
        Document(name='doc_01', text=input.text)
    ]

    model_bilstmcrf = 'src/bilstmcrf.pt'
    model_crf = 'src/crf.pickle'

    if os.path.isfile(model_bilstmcrf) & os.path.isfile(model_crf):
        print('Model exists')
    else:
        return {"output": "Models not found! Contact the administrator."}

    print('Tokenizing...')
    tokenizer = TokenizerFactory().tokenizer(corpus='ehr')

    print('Tagging...')
    tagger_bilstmcrf = BiLSTMCRFTagger(model = model_bilstmcrf, tokenizer = tokenizer, mini_batch_size = 32, verbose = False)
    tagger_crf = CRFTagger(model = model_crf, tokenizer = tokenizer, verbose = False)

    print('Annotating...')
    annotated_docs_bilstmcrf = tagger_bilstmcrf.annotate(document)
    annotated_docs_crf = tagger_crf.annotate(document)

    print(annotated_docs_bilstmcrf[0])
    print(annotated_docs_crf[0])

    return {"bilstmcrf": annotated_docs_bilstmcrf[0], "crf": annotated_docs_crf[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)