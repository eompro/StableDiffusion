import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline
from google.colab import files
from PIL import Image
from random import randint
from accelerate import Accelerator

def sd_imgtoimg_pipeline(token):
    
    device = "cuda"
    accelerator = Accelerator()
    device = accelerator.device
    model_path = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=token
    ).to(device)
    
    return pipe

# Img 2 Img 함수 선언

def sd_imgtoimg_function(prompt, pipe, file_name, strength, seed = None):
    image = Image.open(file_name).convert("RGB").resize((512,512), resample=Image.LANCZOS)

    device = "cuda"

    if seed == None:
        seed_no = randint(1, 999999999)
    else:
        seed_no = seed

    generator = torch.Generator(device=device).manual_seed(seed_no)
    with autocast(device):
        image = pipe(prompt=prompt, init_image=image, strength=strength, guidance_scale=7.5, generator=generator).images[0]

    return image