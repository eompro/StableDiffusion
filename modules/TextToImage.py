import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from random import randint
from accelerate import Accelerator

def sd_texttoimg_pipeline(token):
    device = "cuda"
    accelerator = Accelerator()
    device = accelerator.device

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision = 'fp16', 
        torch_dtype = torch.float16,
        use_auth_token=token
    ).to(device)

    return pipe


def sd_texttoimg_function(pipe, prompt, seed = None):
    device = "cuda"

    if seed == None:
        seed_no = randint(1, 999999999)
    else:
        seed_no = seed

    generator = torch.Generator(device=device).manual_seed(seed_no)
    with autocast(device):
        image = pipe(prompt=prompt, generator=generator)['images'][0]

    return image