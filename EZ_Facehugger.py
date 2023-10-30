import requests
import os
from tqdm import tqdm
import hashlib
import re

# Define the base URL for your Hugging Face repository
base_download_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main"

file_urls = [
    f"{base_download_url}/text_encoder/config.json",   
    f"{base_download_url}/unet/config.json",    
    f"{base_download_url}/vae/config.json",  
    f"{base_download_url}/tokenizer/merges.txt",
    f"{base_download_url}/tokenizer/special_tokens_map.json",
    f"{base_download_url}/tokenizer/tokenizer_config.json",
    f"{base_download_url}/tokenizer/vocab.json",
    f"{base_download_url}/scheduler/scheduler_config.json",
    f"{base_download_url}/model_index.json",
    f"{base_download_url}/safety_checker/pytorch_model.bin",
    f"{base_download_url}/text_encoder/pytorch_model.bin",
    f"{base_download_url}/text_encoder/model.fp16.safetensors",
    f"{base_download_url}/vae/diffusion_pytorch_model.bin",
    f"{base_download_url}/vae/diffusion_pytorch_model.fp16.safetensors",
    f"{base_download_url}/unet/diffusion_pytorch_model.bin", 
    f"{base_download_url}/unet/diffusion_pytorch_model.fp16.safetensors",    
]

local_file_paths = [
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "scheduler/scheduler_config.json",
    "model_index.json",
    "safety_checker/pytorch_model.bin",
    "text_encoder/pytorch_model.bin",
    "text_encoder/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.fp16.safetensors"
]

script_directory = os.path.dirname(os.path.abspath(__file__))
sd_root = "models/StableDiffusion/"

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path, leave=True) as pbar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))

for url, local_path in zip(file_urls, local_file_paths):
    # Determine full local path
    local_path_full = os.path.join(script_directory, sd_root, local_path)
    os.makedirs(os.path.dirname(local_path_full), exist_ok=True)

    should_download = False
    # For bin, safetensors, and ckpt files, perform SHA-256 check
    if local_path_full.endswith(('.bin', '.safetensors', '.ckpt')):
        try:
            with open(local_path_full, 'rb') as file:
                local_sha256 = hashlib.sha256(file.read()).hexdigest()
        except FileNotFoundError:
            print(f"Local file {local_path_full} not found. Redownloading.")
            should_download = True
        else:
            sha256_url = url.replace("/resolve/", "/raw/")
            response = requests.get(sha256_url)

            remote_sha256_response = response.text
            match = re.search(r'sha256:(\w+)', remote_sha256_response)

            if match:
                remote_sha256 = match.group(1)
                if local_sha256 != remote_sha256:
                    print(f"local_sha256 {local_sha256} doesn't match remote {remote_sha256}. Redownloading: {local_path_full}")
                    should_download = True
                else:
                    print(f"Already have {local_path_full}. Skipping...")
            else:
                print(f"Failed to extract SHA-256 hash from response.")
                should_download = True
    else:  # For txt and json files, check file size
        response = requests.head(url)
        remote_size = int(response.headers.get('content-length', 0))
        try:
            local_size = os.path.getsize(local_path_full)
        except FileNotFoundError:
            local_size = 0

        if local_size != remote_size:
            print(f"Local file size {local_size} doesn't match remote size {remote_size}. Redownloading: {local_path_full}")
            should_download = True
        else:
            print(f"Already have{local_path_full}.")

    if should_download:
        try:
            download_file(url, local_path_full)
            print(f"Downloaded: {local_path_full}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
