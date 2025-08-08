import modal
import os
import wget
import shutil


app = modal.App()
volume = modal.Volume.from_name("elabs-phi-verse", create_if_missing=True)

#@app.function(volumes={"/myvol": volume})
@app.local_entrypoint()
def main():
    #g.remote()
    url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf'
    file_Path = 'mistral2.gguf'
    wget.download(url, file_Path)
    with volume.batch_upload() as batch:
        batch.put_file("mistral2.gguf", "/mistral2.gguf")
    

    
    
    
    
