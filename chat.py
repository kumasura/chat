from modal import App, Image, gpu, web_endpoint
import modal
from typing import Any, List, Optional
from pydantic import BaseModel, Field
import time
import uuid

# ------------------------------
# App & Image
# ------------------------------
app = App(name="chatbot")

outlines_image = (
    Image.from_registry(
        "nvidia/cuda:12.3.0-devel-ubuntu22.04", add_python="3.12"
    )
    .run_commands(
        "apt-get update",
        "echo 'tzdata tzdata/Areas select Europe' | debconf-set-selections",
        "echo 'tzdata tzdata/Zones/Europe select London' | debconf-set-selections",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata",
        "apt-get install -y software-properties-common",
        "add-apt-repository ppa:ubuntu-toolchain-r/test",
        "apt-get install -y build-essential git clang graphviz",
        """CMAKE_ARGS="-DLLAMA_CUDA=on -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" pip install llama-cpp-python --no-cache-dir""",
    )
    .pip_install(
        "llama-index",
        "pydantic>=2",
        "fastapi[standard]"
    )
)

# ------------------------------
# OpenAI-compatible Request Model
# ------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "local-llama"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

# ------------------------------
# OpenAI-compatible Endpoint
# ------------------------------
@app.function(
    image=outlines_image,
    gpu=gpu.A100(size="80GB"),
    timeout=300,  # longer timeout for large contexts
    volumes={"/my_vol": modal.Volume.from_name("elabs-phi-verse")}
)
#@web_endpoint(method="POST", label="v1/chat/completions")
@modal.fastapi_endpoint(
    method="POST",
    label="v1-chat-completions"  # ✅ label rules
)
async def v1_chat_completions(request: ChatCompletionRequest) -> Any:
    from llama_cpp import Llama
    from pydantic import BaseModel, Field
    # Load LLaMA model
    llm = Llama(
        model_path="/my_vol/mistral2.gguf",
        n_batch=1024,
        n_threads=10,
        n_gpu_layers=-1,
        n_ctx=32768
    )

    # Convert request to llama-cpp's expected format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Run the model
    output = llm.create_chat_completion(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream
    )

    # Prepare OpenAI-compatible response
    response = {
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

    return response
