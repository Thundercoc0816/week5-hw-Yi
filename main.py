from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Sentiment-API")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

class TextIn(BaseModel):
    text: Union[str, List[str]]

@app.post("/predict")
def predict(payload: TextIn):
    texts = [payload.text] if isinstance(payload.text, str) else payload.text
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=1)
    labels = probs.argmax(dim=1)
    id2label = model.config.id2label
    return [
        {"label": id2label[int(i)], "score": float(p)}
        for i, p in zip(labels, probs.max(dim=1).values)
    ]

@app.get("/")
def root():
    return {"msg": "ok"}