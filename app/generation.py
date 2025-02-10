import os
import torch
from torch.cuda.amp import autocast
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

# We can either import pipeline directly, or load the model + tokenizer ourselves.
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

##################################################################
# 1. Model & Pipeline Setup
##################################################################

# The new model
MODEL_NAME = "ruslanmv/Medical-Llama3-8B"

# Create a text-generation pipeline
# - device_map="auto" tries to load the model onto available GPU
# - torch_dtype=torch.float16 for half-precision on GPU
# - if your GPU memory is limited, you might experiment with load_in_8bit or load_in_4bit (bitsandbytes)
pipe = pipeline(
    "text-generation", model=MODEL_NAME, device_map="auto", torch_dtype=torch.float16
)

# (Optional) If you prefer more direct control:
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


##################################################################
# 2. FastAPI App
##################################################################
app = FastAPI(
    title="GCP LLM Generation Service",
    description="Runs ruslanmv/Medical-Llama3-8B on an NVIDIA L4 GPU with half-precision.",
    version="1.0",
)


##################################################################
# 3. Request/Response Schema
##################################################################
class GenerationRequest(BaseModel):
    context: str
    query: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9


class GenerationResponse(BaseModel):
    answer: str


##################################################################
# 4. Generation Endpoint
##################################################################
@app.post("/generate", response_model=GenerationResponse)
def generate_text(req: GenerationRequest):
    """
    Receives a 'context' + 'query' and returns a short medical answer.
    """
    prompt = build_prompt(req.context, req.query)

    # Inference with autocast for half precision on CUDA
    with torch.no_grad(), autocast(
        device_type="cuda", enabled=torch.cuda.is_available()
    ):
        outputs = pipe(
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            do_sample=True,
            num_return_sequences=1,
        )

    # The pipeline returns a list of generated sequences. We pick the first.
    generated_text = outputs[0]["generated_text"]
    # Optionally, remove the prompt from the final text or do other post-processing.
    answer = generated_text.strip()
    return GenerationResponse(answer=answer)


##################################################################
# 5. Prompt Construction
##################################################################
def build_prompt(context: str, query: str) -> str:
    """
    Combine context + query into a prompt that fits Medical-Llama3-8B's instruction style.
    Keep it short for sub-second latencies.
    """
    return (
        "Below is relevant medical context:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{query}\n\n"
        "Provide a concise, medically accurate answer:\n"
    )


##################################################################
# 6. Server Entry Point
##################################################################
if __name__ == "__main__":
    # Single worker typically recommended for large models
    # so it doesn't get duplicated in memory.
    uvicorn.run(
        "generation:app",
        host="0.0.0.0",
        port=8001,
        workers=1,
    )
