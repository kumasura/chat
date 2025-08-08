from modal import App, Image, Secret, gpu, web_endpoint
import modal
app = App(
    name="chatbot"
)  
outlines_image = Image.from_registry(
    "nvidia/cuda:12.3.0-devel-ubuntu22.04", add_python="3.12").run_commands("apt-get update",
    "echo 'tzdata tzdata/Areas select Europe' | debconf-set-selections","echo 'tzdata tzdata/Zones/Europe select London' | debconf-set-selections",
"DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata","apt-get install -y software-properties-common","add-apt-repository ppa:ubuntu-toolchain-r/test","apt-get install -y build-essential git clang graphviz","""CMAKE_ARGS="-DLLAMA_CUDA=on -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80" pip install llama-cpp-python --no-cache-dir""").pip_install(
    "llama-index",
    "fastapi[standard]"
    )

   
@app.function(image=outlines_image, gpu=gpu.A100(size="80GB"), timeout=60, volumes={"/my_vol": modal.Volume.from_name("elabs-phi-verse")})
@web_endpoint(method="POST")
def chatbot(prompt: str):
    from llama_cpp import Llama
    
    # Create an instance of the Llama class and load the model
    llama_model = Llama("/my_vol/mistral-7b-instruct-v0.2.Q8_0.gguf", n_batch=1024, n_threads=10, n_gpu_layers=-1, n_ctx = 32768)

    response = llm.create_chat_completion(messages=prompt)
    return response["choices"][0]["message"]["content"]
          
   


    
