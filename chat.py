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
        """CMAKE_ARGS="-DLLAMA_CUDA=on -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" pip install llama-cpp-python --no-cache-dir"""
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

# -------------------
# FastAPI App
# -------------------
web_app = FastAPI()

@web_app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    from llama_cpp import Llama

    llm = Llama(
        model_path="/my_vol/mistral2.gguf",
        n_batch=1024,
        n_threads=10,
        n_gpu_layers=-1,
        n_ctx=32768
    )

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
