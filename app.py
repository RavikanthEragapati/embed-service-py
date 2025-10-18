import torch
import numpy as np
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# 1. load model
model_id = '/Users/ravieragapati/MyPOC/mxbai-embed-large-v1/'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).half()
model.eval()

app = FastAPI(title="Embedding Service", version="1.0.0")

def pooling(outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
    # CLS pooling: take the embedding of the first token (CLS)
    if strategy == 'cls':
        outputs = outputs[:, 0]
    # Mean pooling: average all token embeddings, masked by attention mask
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1,
                                                                               keepdim=True)
    else:
        raise NotImplementedError(f"Pooling strategy '{strategy}' not supported.")

    # Detach from graph, move to CPU, and convert to numpy array
    return outputs.detach().cpu().numpy()

# --- Pydantic Models for Request and Response ---
class EmbeddingRequest(BaseModel):
    input: List[str]


class EmbeddingItem(BaseModel):
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingItem]


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    inputs = tokenizer(request.input,
                       padding=True,
                       return_tensors='pt'
                       # truncation = True,
                       # max_length=512,
                       # return_token_type_ids=True,
                       # add_special_tokens=True,
                       # add_special_tokens_parsing=True,
                       )
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state

    embeddings_np = pooling(outputs, inputs, 'cls')

    response_data: List[EmbeddingItem] = []
    for i, embedding in enumerate(embeddings_np):
        response_data.append(
            EmbeddingItem(
                index=i,
                embedding=embedding.tolist()
            )
        )
    return EmbeddingResponse(data=response_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)