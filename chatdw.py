from modal import App, Image
import modal


app = modal.App("Model Downloader")
volume = modal.Volume.from_name("elabs-phi-verse", create_if_missing=True)
outlines_image = (
    Image.from_registry(
        "python:3.13-alpine3.21"
    )
    .run_commands(
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y git",
    )
    .pip_install(
        "torch", 
        "git+https://github.com/huggingface/transformers accelerate",
        "git+https://github.com/huggingface/diffusers"  
    )
)

@app.function(image=outlines_image,memory=1024*64, volumes={"/my_vol": modal.Volume.from_name("elabs-phi-verse")},secrets=[modal.Secret.from_name("huggingface-secret")],)
def download():
    import os
    import torch
    from diffusers import QwenImageEditPipeline
    local_model_path = "./models"
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", cache_dir=local_model_path)
    print("pipeline loaded")
    

if __name__ == "__main__":
    with app.run():
        download.remote()
    print("Downloaded")
    
    
    
    
