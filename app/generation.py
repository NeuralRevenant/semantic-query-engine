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

HF_TOKEN = ""

MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Transformers can authenticate to the gated repo using below
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

##################################################################
# 2. FastAPI App
##################################################################
app = FastAPI(
    title="GCP LLM Generation Service",
    description="Optimized Llama-2 13B for medical QA, referencing doc IDs.",
    version="1.0",
)


##################################################################
# 3. Request/Response Schema
##################################################################
class GenerationRequest(BaseModel):
    context: str
    query: str
    max_new_tokens: int = 256
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
    We do NOT reduce max_new_tokens, but we do truncate the context from the back if needed.
    """
    # 1) Build the full initial prompt
    raw_prompt = build_prompt(req.context, req.query)

    # 2) Tokenize the entire prompt
    tokenized = tokenizer(raw_prompt, return_tensors="pt")
    prompt_length = tokenized.input_ids.shape[1]

    # 3) Compute total requested length
    total_requested = prompt_length + req.max_new_tokens
    if total_requested > 4096:
        # We need to make room for new tokens by truncating the back of the context
        max_prompt_length = 4096 - req.max_new_tokens
        if max_prompt_length < 1:
            max_prompt_length = 1  # Ensure at least 1 token remains

        # 4) Keep only the FIRST max_prompt_length tokens and remove the rest
        trimmed_ids = tokenized.input_ids[0][:max_prompt_length]
        # Convert back to string
        trimmed_prompt_str = tokenizer.decode(trimmed_ids, skip_special_tokens=False)

        print(
            f"[Warning] Original prompt had {prompt_length} tokens, truncated to {max_prompt_length} to allow {req.max_new_tokens} new tokens."
        )
        final_prompt = trimmed_prompt_str
        prompt_length = max_prompt_length  # update
    else:
        final_prompt = raw_prompt

    # 5) Our final max_length ensures we do not exceed 4096 tokens in total
    max_length = prompt_length + req.max_new_tokens

    # Use autocast in half-precision if CUDA is available
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        outputs = pipe(
            final_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,  # Avoid pad token warnings
        )

    generated_text = outputs[0]["generated_text"]
    print(f"Generated Text: {generated_text}")

    answer = extract_model_answer(generated_text, final_prompt)
    print(f"Answer Text: {answer}")

    return GenerationResponse(answer=answer.strip())


##################################################################
# 5. Prompt Helpers
##################################################################
def build_prompt(context: str, query: str) -> str:
    """
    Build an instruction-style prompt with context & query.
    """
    return (
        "You are a helpful AI assistant. Use the provided documents' context or information while following every instruction extremely carefully!\n\n"
        f"### Context:\n{context}\n\n"
        # f"### User Query:\n{query}\n\n"
        "### Response:\n"
    )


def extract_model_answer(generated_text: str, prompt: str) -> str:
    """
    Remove the prompt from the generated text and keep only the model's answer.
    """
    split_marker = "### Response:"
    if split_marker in generated_text:
        return generated_text.split(split_marker, 1)[-1]
    else:
        if prompt in generated_text:
            return generated_text.replace(prompt, "")
        return generated_text


##################################################################
# 6. Server Entry Point
##################################################################
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
    uvicorn.run(
        "generation:app",
        host="0.0.0.0",
        port=8001,
        workers=1,
    )
