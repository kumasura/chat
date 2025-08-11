from modal import App, Image
import modal
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI
import time, uuid

# -------------------
# Modal App + Image
# -------------------
app = App(name="chatbot")

outlines_image = (
    Image.from_registry(
        "nvidia/cuda:12.3.0-devel-ubuntu22.04", add_python="3.12"
    )
    .run_commands(
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata",
        "apt-get install -y software-properties-common build-essential git clang graphviz",
        """CMAKE_ARGS="-DLLAMA_CUDA=on -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" pip install "llama-cpp-python>=0.2.75" --no-cache-dir"""
    )
    .pip_install(
        "llama-index",
        "pydantic>=2",
        "fastapi[standard]"
    )
)

# -------------------
# Request schema
# -------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "local-llama"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

# ---- Token count schema ----
class TokenCountRequest(BaseModel):
    messages: List[ChatMessage]
    add_generation_prompt: bool = True  # same default as create_chat_completion

class TokenCountResponse(BaseModel):
    prompt_tokens: int
    text_preview: Optional[str] = None  # helpful for debugging (first 200 chars)


# -------------------
# FastAPI App
# -------------------
web_app = FastAPI()

# ---- Llama singleton ----
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    from llama_cpp import Llama
    return Llama(
        model_path="/my_vol/mistral2.gguf",
        n_batch=1024,
        n_threads=10,
        n_gpu_layers=-1,
        n_ctx=32768
    )

def render_mistral_chat(messages, add_generation_prompt=True):
    """
    Minimal Mistral-style chat template.
    Supports: system (optional), user, assistant turns.
    Produces the same structure llama.cpp expects for Mistral-like models.
    """
    sys_txt = ""
    parts = []
    first_user_done = False

    # Extract first system message if present
    for m in messages:
        if m["role"] == "system":
            sys_txt = m["content"]
            break

    # Build turns
    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "system":
            # already captured; skip in sequence
            continue

        if role == "user":
            if not first_user_done:
                # First user message may include <<SYS>> block
                if sys_txt:
                    parts.append(
                        f"<s>[INST] <<SYS>>\n{sys_txt}\n<</SYS>>\n\n{content} [/INST]"
                    )
                else:
                    parts.append(f"<s>[INST] {content} [/INST]")
                first_user_done = True
            else:
                parts.append(f"<s>[INST] {content} [/INST]")

        elif role == "assistant":
            # Assistant follows a completed [INST] block
            parts.append(f"{content}</s>")

        # (You can add tool/tool_calls handling here if you need it later)

    # If we’re going to generate, we end after an [INST] block with no assistant yet.
    if add_generation_prompt:
        if not parts or parts[-1].endswith("</s>"):
            # If last was assistant, open a fresh user turn for generation (rare)
            parts.append("<s>[INST] [/INST]")
        # Else the last part is an [INST] … [/INST] block already, which is fine.

    return "\n".join(parts)


@web_app.post("/v1/tokens", response_model=TokenCountResponse)
async def count_tokens(req: TokenCountRequest) -> TokenCountResponse:
    """
    Returns the number of input tokens the model will see for the given chat messages.
    Uses llama.cpp's tokenizer + chat template for *your* GGUF, so it matches generation.
    """
    llm = get_llm()
    # Convert Pydantic models → dicts (role/content), same as your /v1/chat/completions

    role_map = {
    "human": "user",
    "ai": "assistant",
    "system": "system"
    }
    
    msgs = [{"role": role_map[m.type], "content": m.content} for m in req.messages]
    #msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    rendered = render_mistral_chat(
        msgs,
        add_generation_prompt=req.add_generation_prompt
    )

    tokens = llm.tokenize(rendered.encode("utf-8"), add_bos=True)
    return TokenCountResponse(
        prompt_tokens=len(tokens),
        text_preview=rendered[:200]
    )


@web_app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    from llama_cpp import Llama

    llm = get_llm()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    output = llm.create_chat_completion(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": output["choices"][0]["message"],
                "finish_reason": output["choices"][0].get("finish_reason", "stop")
            }
        ],
        "usage": output.get("usage", {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        })
    }

# -------------------
# Mount in Modal
# -------------------
@app.function(
    image=outlines_image,
    gpu="A100-80GB",
    timeout=300,
    volumes={"/my_vol": modal.Volume.from_name("elabs-phi-verse")}
)
@modal.asgi_app()
def fastapi_app():
    return web_app
