from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModel, AutoTokenizer
import torch

# 1. load model
model_id = '/Users/ravieragapati/MyPOC/mxbai-embed-large-v1/'
tokenize = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).half()
model.eval()

app = FastAPI()

class EmbedRequest(BaseModel):
    input: list[str]

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbedRequest):
    inputs = tokenize(request.input,
                      padding=True,
                      return_tensors="pt",
                      # truncation = True,
                      # max_length=512,
                      # return_token_type_ids=True,
                      # add_special_tokens=True,
                      # add_special_tokens_parsing=True,
                      )
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    data = [
        {"index": i, "embedding": emb.tolist()}
        for i, emb in enumerate(embeddings)
    ]
    return {"data": data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)