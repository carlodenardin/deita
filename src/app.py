from fastapi import FastAPI
from pydantic import BaseModel

class Input(BaseModel):
    text: str

app = FastAPI()

@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}

@app.post("/api/python/ehr")
def ehr(input: Input):
    return {"output": input.text}
