import requests
import os
from tqdm import tqdm
import hashlib
import re

# Define the base URL for your Hugging Face repository
base_download_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5"

file_urls = [
    f"{base_download_url}/raw/main/text_encoder/config.json",
    f"{base_download_url}/resolve/main/text_encoder/model.fp16.safetensors",
    f"{base_download_url}/raw/main/unet/config.json",
    f"{base_download_url}/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
    f"{base_download_url}/raw/main/vae/config.json",
    f"{base_download_url}/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
    f"{base_download_url}/raw/main/tokenizer/merges.txt",
    f"{base_download_url}/raw/main/tokenizer/special_tokens_map.json",
    f"{base_download_url}/raw/main/tokenizer/tokenizer_config.json",
    f"{base_download_url}/raw/main/tokenizer/vocab.json",
    f"{base_download_url}/raw/main/model_index.json",
    f"{base_download_url}/raw/main/scheduler/scheduler_config.json"
]

script_directory = os.path.dirname(os.path.abspath(__file__))


local_file_paths = [
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "model_index.json",
    "scheduler_config.json"
]

sd_root = "models/StableDiffusion/"

for url, local_path in zip(file_urls, local_file_paths):
    if not local_path.endswith('.safetensors'):
        file_url = url
        local_path = os.path.join(script_directory, sd_root, local_path)

        # Check if the file already exists locally and has the correct size
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            response = requests.head(file_url)
            remote_size = int(response.headers.get('content-length', 0))

            # Compare file sizes
            if local_size == remote_size:
                print(f"File already exists with correct size: {local_path}")
                continue

        # Download non-.safetensors files
        response = requests.get(file_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        try:
            with open(local_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path, leave=True) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        pbar.update(len(data))
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")
        else:
            print(f"Downloaded: {local_path}")
    else:
        file_url = url
        sha256_url = file_url.replace("/resolve/", "/raw/")
        local_path = os.path.join(script_directory, sd_root, local_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Check if the file already exists locally
    if os.path.exists(local_path):
        local_sha256 = hashlib.sha256(open(local_path, 'rb').read()).hexdigest()
        response = requests.get(sha256_url)
        remote_sha256_response = response.text

        # Extract SHA-256 hash using regular expression
        match = re.search(r'sha256:(\w+)', remote_sha256_response)
        if match:
            remote_sha256 = match.group(1)
        else:
            print(f"Failed to extract SHA-256 hash from response.")
            continue

        # Compare SHA-256 hashes
        if local_sha256 == remote_sha256:
            print(f"SHA check passed: {local_path}")
            continue
        else:
            print(f"local_sha256 {local_sha256} doesn't match remote {remote_sha256} redownloading: {local_path}")
    else:
        print(f"Downloading: {file_url}")

    response = requests.get(file_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    try:
        with open(local_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path, leave=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    pbar.update(len(data))
    except Exception as e:
        print(f"Failed to download {file_url}: {e}")
    else:
        print(f"Downloaded: {local_path}")
