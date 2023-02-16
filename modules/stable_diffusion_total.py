import numpy as np
import torch
import getpass
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from random import randint
from accelerate import Accelerator


class Text_To_Image :
    
    def __init__(self, token, prompt, seed):
        self.token = token
        self.prompt = prompt
        self.seed = seed
        pass
    
    # Text To Image
    def sd_texttoimg_pipeline(self, token):
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
    
    def sd_texttoimg_function(self, pipe, prompt, seed):
        device = "cuda"

        if seed == "":
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=prompt, generator=generator)['images'][0]

        return image


class Image_To_Image :
    
    def __init__(self, token, file_name, prompt, strength, seed):
        self.token = token
        self.file_name = file_name
        self.prompt = prompt
        self.strength = strength
        self.seed = seed
        
    
    # Text To Image
    def sd_imgtoimg_pipeline(self, token):
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=token
        ).to(device)
        
        return pipe
    
    def sd_imgtoimg_function(self, pipe, prompt, file_name, strength, seed):
        image = Image.open(file_name).convert("RGB").resize((512,512), resample=Image.LANCZOS)

        device = "cuda"

        if seed == "":
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5, generator=generator).images[0]

        return image
    
###################################################################################    

class Image_Extend:
    
    def __init__(self, token, prompt, file_name, a, b, output_name, seed):
        self.token = token
        self.prompt = prompt
        self.file_name = file_name
        self.a = a
        self.b = b
        self.output_name = output_name
        self.seed = seed
    
    def sd_extend_pipeline(self, token):
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        model_id = "runwayml/stable-diffusion-inpainting"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=token
        ).to(device)
        
        return pipe
    
    def sd_extend_crop_mask(self, file_name, a, b):
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
    
    def sd_extend_result_img(self, pipe, prompt, extend_img, image, mask_image, a, b, seed):
        num_samples = 1
        if seed == "":
            seed = randint(0,9999999999)
        else:
            seed = int(seed)
            
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        generator = torch.Generator(device=device).manual_seed(seed) # change the seed to get different results

        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=7.5,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images[0]

        extend_img_array = np.array(extend_img)
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
    
    def sd_extend_function(self, pipe, file_name, prompt, a, b, output_name = "", seed = ""):
        
        extend_img, image, mask_image = self.sd_extend_crop_mask(file_name, a, b)
        

        final_result = self.sd_extend_result_img(pipe, prompt, extend_img, image, mask_image, a, b, seed)
            
        if output_name == "":
            return final_result
        else:
            final_result.save(output_name, output_name.split(".")[-1])
            return final_result

    





def text_to_image():
    
    print('Input the Huggingface Token: ')
    Huggingface_Token = getpass.getpass('')
    token=Huggingface_Token

    # Session Name
    print('Input the prompt: ')
    prompt = input('')

    print('Input the prompt: ')
    print('If you want Random Seed, input Nothing.')
    seed = input('')
    
    diffusion = Text_To_Image(token, prompt, seed)
    
    try:
        image = diffusion.sd_texttoimg_function(pipe_t2i, prompt, seed)
    except:
        pipe_t2i = diffusion.sd_texttoimg_pipeline(token)
        image = diffusion.sd_texttoimg_function(pipe_t2i, prompt, seed)
        
    return image


def image_to_image():
    
    print('Input the Huggingface Token: ')
    Huggingface_Token = getpass.getpass('')
    token=Huggingface_Token

    print('Input the file_name(or file_path) of image: ') 
    file_name = input('')

    print('Input the prompt: ')
    prompt = input('')
    
    print('Input the strength: ')
    print('Strength is recommended between 0.4 and 0.6.')
    strength = float(input(''))
    
    print('Input the seed: ')
    print('If you want Random Seed, input Nothing.')
    seed = input('')
    
    diffusion = Image_To_Image(token, file_name, prompt, strength, seed)
    
    try:
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
    except:
        pipe_i2i = diffusion.sd_imgtoimg_pipeline(token)
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
        
    return image


def image_extend():
    print('Input the Huggingface Token: ')
    Huggingface_Token = getpass.getpass('')
    token=Huggingface_Token

    print('Input the file_name(or file_path) of image: ') 
    file_name = input('')

    print('Input the prompt: ')
    prompt = input('')
    
    print('Input the the x,y coordinates of the upper left vertex (ex. 325 410): ')
    num_list = list(map(int,input('').split( )))
    a, b = num_list[0], num_list[1]
    
    print('Input the Output Name: ')
    print("If you don't want save the result image, input Nothing.")
    output_name = input('')
    
    print('Input the prompt: ')
    print('If you want Random Seed, input Nothing.')
    seed = input('')
    
    diffusion = Image_Extend(token, file_name, prompt, a, b, output_name, seed)
    
    try:
        image = diffusion.sd_extend_function(pipe_ie, file_name, prompt, a, b, output_name, seed)
    except:
        pipe_ie = diffusion.sd_extend_pipeline(token)
        image = diffusion.sd_extend_function(pipe_ie, file_name, prompt, a, b, output_name, seed)
        
    return image