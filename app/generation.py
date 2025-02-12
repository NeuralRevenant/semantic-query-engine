import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

##################################################################
# 1. Model & Pipeline Setup
##################################################################

MODEL_NAME = "ruslanmv/Medical-Llama3-8B"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

# Using the default pipeline.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

##################################################################
# 2. FastAPI App
##################################################################
app = FastAPI(
    title="GCP LLM Generation Service",
    description="Optimized Medical LLM with concise output, referencing doc IDs as needed.",
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
    Receives a 'context' + 'query' and returns a short medical answer
    that cites relevant doc IDs without returning the entire prompt.
    """
    prompt = build_prompt(req.context, req.query)

    # Use autocast in half-precision for performance
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        outputs = pipe(
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            do_sample=True,
            num_return_sequences=1,
        )

    generated_text = outputs[0]["generated_text"]

    print(f"Generated Text: {generated_text}")
    # Post-processing to remove the prompt or repeated text:
    # We look for the "### Response:" delimiter below and extract just that portion.
    answer = extract_model_answer(generated_text, prompt)
    print(f"Answer Text: {answer}")

    return GenerationResponse(answer=answer.strip())


##################################################################
# 5. Improved Prompt Construction
##################################################################
def build_prompt(context: str, query: str) -> str:
    """
    Build an instruction-style prompt that:
    - Provides user context with doc IDs.
    - Asks for a concise answer that references only relevant doc IDs.
    - Prohibits returning the entire prompt or chain of thought.
    """
    return (
        f"### User Query (or Question): {query}\n\n"
        f"### Relevant Information/Context along with the instructions below:\n{context}"
    )


def extract_model_answer(generated_text: str, prompt: str) -> str:
    """
    A utility to remove the entire prompt portion from the generated text,
    returning only the model's final portion after '### Response:'.
    """
    # Split off everything before '### Response:' if it still appears.
    # If the model doesn't strictly follow that might need more robust logic.
    split_marker = "### Response:"
    if split_marker in generated_text:
        # Get only what comes after '### Response:'.
        return generated_text.split(split_marker, 1)[-1]
    else:
        # Fallback: remove the initial prompt from the output if present
        if prompt in generated_text:
            return generated_text.replace(prompt, "")
        
        return generated_text


##################################################################
# 6. Server Entry Point
##################################################################
if __name__ == "__main__":
    # Tweak CUDA memory allocation behavior if needed
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"

    uvicorn.run(
        "generation:app",
        host="0.0.0.0",
        port=8001,
        workers=1,
    )
