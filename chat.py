from modal import App, Image, Secret, gpu, web_endpoint
import modal
from typing import Any, List, Optional
from pydantic import BaseModel, Field
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
# Request Models
# ------------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role (user/assistant/system)")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default="local-llama", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(default=False, description="Stream response or not")


# ------------------------------
# API Function
# ------------------------------
@app.function(
    image=outlines_image,
    gpu=gpu.A100(size="80GB"),
    timeout=60,
    volumes={"/my_vol": modal.Volume.from_name("elabs-phi-verse")}
)
@web_endpoint(method="POST")
async def v1_chat_completions(request: ChatCompletionRequest) -> Any:
    from llama_cpp import Llama
    from pydantic import BaseModel, Field
    # Load model
    llm = Llama(
        model_path="/my_vol/mistral2.gguf",
        n_batch=1024,
        n_threads=10,
        n_gpu_layers=-1,
        n_ctx=32768
    )

    # Convert to llama-cpp format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Generate completion
    completion = llm.create_chat_completion(
        messages=messages,
        stream=request.stream
    )

    # Return in OpenAI-compatible format
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": completion["choices"][0]["message"],
                "finish_reason": completion["choices"][0].get("finish_reason", "stop")
            }
        ]
    }
