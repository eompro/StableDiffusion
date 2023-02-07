import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from random import randint
from accelerate import Accelerator


# 크롭이미지, 마스크 만드는 함수
## 원본 이미지와 인페인트 할 512*512 사각형의 좌상단 좌표 (a,b)
def sd_extend_crop_mask(file_name, a, b):
    main_img = Image.open(file_name).convert("RGBA")

    main_width, main_height = main_img.size

    extend_width = main_width + (512 * 2)
    extend_height = main_height + (512 * 2)
    extend_square_w = np.full((extend_height, extend_width, 4), (255, 255, 255, 0), dtype=np.uint8)

    main_array = np.array(main_img)
    for width in range(0, main_width):
        for height in range(0, main_height):
            extend_square_w[height+512][width+512] = main_array[height][width]

    extend_main_img = Image.fromarray(extend_square_w)

    # crop extend_main_img
    extend_crop = extend_main_img.crop((a,b,a+512,b+512))
    extend_crop

    # a, b value 검증
    crop_array = np.array(extend_crop)
    zero_count = crop_array[:,:,3].reshape(-1).tolist().count(0)
    if zero_count == 0:
        print("a,b 값 다시 설정 필요.")
        return

    # 5. crop_array와 투명도를 이용하여 마스크 생성
    mask_array = crop_array.copy()
    for i in range(512):
        for j in range(512):
            if mask_array[i][j][3] == 255:
                mask_array[i][j] = [0,0,0,255]
            else:
                mask_array[i][j] = [255,255,255,255]
    mask = Image.fromarray(mask_array)

    return extend_main_img, extend_crop, mask

def sd_extend_pipeline(token):
    device = "cuda"
    accelerator = Accelerator()
    device = accelerator.device
    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=token
    ).to(device)
    
    return pipe

def sd_extend_result_img(pipe, prompt, negative_prompt, guidance_scale, extend_img, image, mask_image, a, b, seed):
    num_samples = 1
    if seed == None:
        seed = randint(0,9999999999)
    else:
        seed = seed
        
    device = "cuda"
    accelerator = Accelerator()
    device = accelerator.device
    generator = torch.Generator(device=device).manual_seed(seed) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        negative_prompt = negative_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images[0]

    extend_img_array = np.array(extend_img)
    # images.convert("RGBA")
    images_array = np.array(images.convert("RGBA"))
    for i in range(512):
        for j in range(512):
            extend_img_array[b+i][a+j] = images_array[i][j]

    for_crop_h, for_crop_w = extend_img_array.shape[:2]

    w_list, h_list = [], []

    for h in range(for_crop_h):
        for w in range(for_crop_w):
            pixel = extend_img_array[h][w][3]
            if pixel == 255:
                w_list.append(w)
                h_list.append(h)

    result_img = Image.fromarray(extend_img_array)
    final_crop = result_img.crop((min(w_list),min(h_list),max(w_list),max(h_list)))
    return final_crop

def sd_extend_function(pipe, file_name, prompt, negative_prompt, a, b, output_name, guidance_scale = 7.5, seed = None):
    extend_img, image, mask_image = sd_extend_crop_mask(file_name, a, b)

    final_result = sd_extend_result_img(pipe, prompt, negative_prompt, guidance_scale, extend_img, image, mask_image, a, b, seed)
    final_result.save(output_name, output_name.split(".")[-1])
    return final_result