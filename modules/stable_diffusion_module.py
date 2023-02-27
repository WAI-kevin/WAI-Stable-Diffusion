import numpy as np
import torch
import getpass
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from random import randint
from accelerate import Accelerator
import os
import argparse


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

        print("prompt : ", prompt)
        print("seed : ", seed_no)
        
        output_path = os.getcwd() + "/TexttoImage"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        image.save(output_path + f"/T2I_{prompt}_{seed_no}.png", "png")
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

        if seed == "" or seed == None:
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5, generator=generator).images[0]
        
        print("kr_prompt : ", prompt)    
        print("seed : ", seed_no)
        
        output_path = os.getcwd() + "/ImagetoImage"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        image.save(output_path + f"/I2I_{prompt}_{seed_no}.png", "png")
        return image
    
def image_to_image(token, prompt, file_name, strength, seed):
    
    diffusion = Image_To_Image(token, file_name, prompt, strength, seed)
    
    try:
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
    except:
        pipe_i2i = diffusion.sd_imgtoimg_pipeline(token)
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
        
    return image


class Image_Extend:
    
    def __init__(self, token, prompt, file_name, a, b, seed):
        self.token = token
        self.prompt = prompt
        self.file_name = file_name
        self.a = a
        self.b = b
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
        if seed == "" or seed == None:
            seed_no = randint(0,9999999999)
        else:
            seed_no = int(seed)
            
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        generator = torch.Generator(device=device).manual_seed(seed_no) # change the seed to get different results

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
        
        print("prompt : ", prompt)    
        print("seed : ", seed_no)
        
        output_path = os.getcwd() + "/ImageExtend"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        final_crop.save(output_path + f"/IEx_{prompt}_{seed_no}.png", "png")
        return final_crop
    
    def sd_extend_function(self, pipe, file_name, prompt, a, b, seed = ""):
        
        extend_img, image, mask_image = self.sd_extend_crop_mask(file_name, a, b)
        

        final_result = self.sd_extend_result_img(pipe, prompt, extend_img, image, mask_image, a, b, seed)
            

        return final_result


# 코드 수정 작업 진행중

class FineTuning:
    
    def __init__(self, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, INSTANCE_DIR):
        self.UNet_Training_Steps = UNet_Training_Steps, 
        self.UNet_Learning_Rate = UNet_Learning_Rate,
        self.Text_Encoder_Training_Steps = Text_Encoder_Training_Steps,
        self.Text_Encoder_Learning_Rate = Text_Encoder_Learning_Rate, 
        self.Session_Name = Session_Name
        self.WORKSPACE='/content/Fast-Dreambooth'
        self.OUTPUT_DIR="/content/models/"+ Session_Name
        self.SESSION_DIR=self.WORKSPACE+'/Sessions/'+ Session_Name
        self.INSTANCE_DIR=INSTANCE_DIR
        self.MODEL_NAME="/content/stable-diffusion-v1-5"
        self.PT=""
        pass
    
    # Line Logging
    def line_logging(self, *messages):
        import datetime
        import sys
        today = datetime.datetime.today()
        log_time = today.strftime('[%Y/%m/%d %H:%M:%S]')
        log = []
        for message in messages:
            log.append(str(message))
        print(log_time + '::' + ' '.join(log) + '')
        sys.stdout.flush()
    
    
    # Environment Setting
    def sd_custom_environment(self):
        import os
        import subprocess
        import shutil
        import glob
        from distutils.dir_util import copy_tree
        import time
        self.line_logging("Start Env. Setting")
        os.chdir('/content/')
        subprocess.run(['pip', 'install', '-q', '--no-deps', 'accelerate==0.12.0'])
        subprocess.call (['wget', '-q', '-i', '/', "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/dbdeps.txt"])

        f = open('/content/dbdeps.txt')
        lines = f.readlines()
        for i in range(len(lines)):
            subprocess.call (['wget', '-q', '/', f"https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/deps.{i+1}"])
        for i in range(len(lines)):
            try:
                shutil.move(f"deps.{i+1}", f"deps.zip.00{i+1}")
            except:
                pass

        cmd = ['7z', 'x', 'deps.zip.001']
        sp = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        time.sleep(20)

        file_source = '/content/usr/local/lib/python3.8/dist-packages'
        file_destination = '/usr/local/lib/python3.8/dist-packages'
        copy_tree(file_source, file_destination)
        time.sleep(20)
        
        shutil.rmtree('/content/usr')

        file_list = []
        file_list.extend(glob.glob("*.00*"))
        file_list.extend(glob.glob("*.txt"))
        for file_name in file_list:
            os.remove(file_name)
        subprocess.run(["git", "clone",  "https://github.com/TheLastBen/diffusers", '--depth=1', '--branch=updt'])
        self.line_logging('Done, proceed')
        
    # create Session
    def sd_custom_create_session(self, MODEL_NAME, SESSION_DIR, INSTANCE_DIR):
        import os
        if os.path.exists(str(SESSION_DIR)):
            self.line_logging('Loading session with no previous model, using the original model or the custom downloaded model')
            if MODEL_NAME=="":
                self.line_logging('No model found, use the "Model Download" cell to download a model.')
            else:
                self.line_logging('Session Loaded, proceed to uploading instance images')


        elif not os.path.exists(str(SESSION_DIR)):
            # %mkdir -p "$INSTANCE_DIR"
            os.makedirs(SESSION_DIR)
            self.line_logging('Creating session...')
            if MODEL_NAME=="":
                self.line_logging('No model found, use the "Model Download" cell to download a model.')
            else:
                self.line_logging('Session created, proceed to uploading instance images')
                
    # upload_image_replace
    # 이미지, 캡션에 들어간 띄어쓰기를 "-" 로 바꿔주는 함수
    def sd_custom_upload_image_replace(self, directory):
        import shutil
        import glob

        inst_list = glob.glob(directory+"/*")
        for i in inst_list:
            old_name = i.split("/")[-1]
            new_name = old_name.replace(" ", "-")
            shutil.move(directory +"/"+old_name, directory + "/" + new_name)
            
    # Upload Image
    def sd_custom_upload_image(self, SESSION_DIR, INSTANCE_DIR, IMAGE_DIR):
        import shutil
        import os
        from glob import glob
        from tqdm import tqdm
        
        self.line_logging("Start : Upload Image...")

        if not os.path.exists(str(INSTANCE_DIR)):
            os.makedirs(INSTANCE_DIR)

        if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
            shutil.rmtree(str(INSTANCE_DIR) + "/.ipynb_checkpoints")
            

        # up=""  
        # uploaded = files.upload()
        
        # # 캡션과 이미지 파일 분리
        # for filename in uploaded.keys():
        #     if filename.split(".")[-1]=="txt":
        #         shutil.move(filename, CAPTIONS_DIR)
        #     up=[filename for filename in uploaded.keys() if filename.split(".")[-1]!="txt"]
            
        # 이미지 파일들 INST_DIR로 이동, bar_Format은 막대기 모양인듯
        

        
        #d 이미지, 캡션 파일 이름의 빈칸을 "-"로 바꿔줌
        for directory in [INSTANCE_DIR]:
            self.sd_custom_upload_image_replace(directory)  
        
        # 파일 압축  
        os.chdir(SESSION_DIR)
        if os.path.exists("instance_images.zip"):
            os.remove("instance_images.zip")
            
        # if os.path.exists("captions.zip"):
        #     os.remove("captions.zip")
        
        shutil.make_archive('instance_images', 'zip', './instance_images')
        # shutil.make_archive('captions', 'zip', './captions')
        
        self.line_logging("Done : Upload Image...")
        
    # Model Download (Ver 1.5)
    def sd_custom_model_download(self,):
        self.line_logging("Start : Model Download...")
        
        import shutil
        import os

        if os.path.exists('/content/stable-diffusion-v1-5'):
            shutil.rmtree('/content/stable-diffusion-v1-5')

        os.chdir('/content')
        os.mkdir('/content/stable-diffusion-v1-5')
        os.chdir('/content/stable-diffusion-v1-5')
        os.system('git init')
        os.system('git lfs install --system --skip-repo')
        os.system('''git remote add -f origin  "https://huggingface.co/runwayml/stable-diffusion-v1-5"''')
        os.system("git config core.sparsecheckout true")
        os.system('''echo -e "\nscheduler\ntext_encoder\ntokenizer\nunet\nfeature_extractor\nsafety_checker\nmodel_index.json\n!*.safetensors" > .git/info/sparse-checkout''')
        os.system("git pull origin main")
        if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
            os.system('''git clone "https://huggingface.co/stabilityai/sd-vae-ft-mse"''')
            os.system('''mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae''')
            os.system("rm -r /content/stable-diffusion-v1-5/.git")
            os.chdir("/content/stable-diffusion-v1-5")
            os.system('''sed -i 's@"clip_sample": false@@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json''')
            os.system('''sed -i 's@"trained_betas": null,@"trained_betas": null@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json''')
            os.system('''sed -i 's@"sample_size": 256,@"sample_size": 512,@g' /content/stable-diffusion-v1-5/vae/config.json''')
            os.chdir("/content")

            self.line_logging('DONE : Model Download...')
        else:
            while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
                self.line_logging('Model Download : Something went wrong')
                
    # TextEnc, UNet Training
    def sd_custom_training(self, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, SESSION_DIR, INSTANCE_DIR, OUTPUT_DIR, Session_Name, PT):    
        self.line_logging("Start : Fine Tuning")
        import random
        import os
        import shutil

        MODELT_NAME = MODEL_NAME
        
        # UNet
        UNet_Training_Steps=UNet_Training_Steps 
        UNet_Learning_Rate = UNet_Learning_Rate
        untlr=UNet_Learning_Rate

        # Text_Encoder
        Enable_text_encoder_training= True
        Text_Encoder_Training_Steps=Text_Encoder_Training_Steps
        Text_Encoder_Learning_Rate = Text_Encoder_Learning_Rate #param ["2e-6", "1e-6","8e-7","6e-7","5e-7","4e-7"] {type:"raw"}
        stptxt=Text_Encoder_Training_Steps
        txlr=Text_Encoder_Learning_Rate

        # Seed
        Seed=""
        if Seed =='' or Seed=='0':
            Seed=random.randint(1, 999999)
        else:
            Seed=int(Seed)
            
        trnonltxt=""
        extrnlcptn=""
        Style=""
        Res = 512

        prec="fp16"
        precision=prec
        GC="--gradient_checkpointing"

        stp=0
        Start_saving_from_the_step=0
        stpsv=Start_saving_from_the_step


        dump_only_textenc = f"""accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
            {trnonltxt} \
            --image_captions_filename \
            --train_text_encoder \
            --dump_only_text_encoder \
            --pretrained_model_name_or_path="{MODELT_NAME}" \
            --instance_data_dir="{INSTANCE_DIR}" \
            --output_dir="{OUTPUT_DIR}" \
            --instance_prompt="{PT}" \
            --seed={Seed} \
            --resolution=512 \
            --mixed_precision={precision} \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 {GC} \
            --use_8bit_adam \
            --learning_rate={txlr} \
            --lr_scheduler="polynomial" \
            --lr_warmup_steps=0 \
            --max_train_steps={stptxt}
            """

        train_only_unet = f"""accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
            {Style} \
            {extrnlcptn} \
            --stop_text_encoder_training={stptxt} \
            --image_captions_filename \
            --train_only_unet \
            --save_starting_step={stpsv} \
            --save_n_steps={stp} \
            --Session_dir="{SESSION_DIR}" \
            --pretrained_model_name_or_path="{MODELT_NAME}" \
            --instance_data_dir="{INSTANCE_DIR}" \
            --output_dir="{OUTPUT_DIR}" \
            --captions_dir="" \
            --instance_prompt={PT} \
            --seed={Seed} \
            --resolution={Res} \
            --mixed_precision={precision} \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 {GC} \
            --use_8bit_adam \
            --learning_rate={untlr} \
            --lr_scheduler="polynomial" \
            --lr_warmup_steps=0 \
            --max_train_steps={UNet_Training_Steps}
            """
        os.chdir('/content')
        # Text Encoder Training
        if Enable_text_encoder_training :
            self.line_logging('Training the text encoder...')
            if os.path.exists(OUTPUT_DIR+'/'+'text_encoder_trained'):
                shutil.rmtree(OUTPUT_DIR+'/'+'text_encoder_trained')
            os.system(dump_only_textenc)

        # UNet Training
        if UNet_Training_Steps!=0:
            self.line_logging('Training the UNet...')
            os.system(train_only_unet)

        # Copy feature_extractor, safety_checker, model_index.json 슈틸 3형제
        try:
            shutil.copytree("/content/stable-diffusion-v1-5/feature_extractor", OUTPUT_DIR + "/feature_extractor")
        except:
            print(f"File exists: '/content/models/{Session_Name}/feature_extractor'")
        try:    
            shutil.copytree("/content/stable-diffusion-v1-5/safety_checker", OUTPUT_DIR + "/safety_checker")
        except:
            print(f"File exists: '/content/models/{Session_Name}/safety_checker'")

        shutil.copyfile('/content/stable-diffusion-v1-5/model_index.json', OUTPUT_DIR + "/model_index.json")
        self.line_logging("Done : FineTuning...")
    # Total Function
    
    def sd_custom_function(self, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, IMAGE_DIR):
        import os
        WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'
        OUTPUT_DIR="/content/models/"+ Session_Name
        SESSION_DIR=WORKSPACE+'/Sessions/'+ Session_Name
        INSTANCE_DIR=SESSION_DIR+'/instance_images'
        MODEL_NAME="/content/stable-diffusion-v1-5"
        PT=""
        ### 1. Environment Setting
        # try:
        #     import wget
        # except:
        #     self.sd_custom_environment()

        ### 2. Create Session
        self.sd_custom_create_session(MODEL_NAME, SESSION_DIR, INSTANCE_DIR)

        ### 3. Image Upload
        self.sd_custom_upload_image(SESSION_DIR, INSTANCE_DIR, IMAGE_DIR)

        ### 4. Model Download (진행중)
        if not os.path.exists('/content/stable-diffusion-v1-5'):
            self.sd_custom_model_download()
        else:
            print("The v1.5 model already exists, using this model.") 

        ### 5. Training
        self.sd_custom_training(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, SESSION_DIR, INSTANCE_DIR, OUTPUT_DIR, Session_Name, PT)


def text_to_image(token, prompt, seed):

    diffusion = Text_To_Image(token, prompt, seed)
    
    try:
        image = diffusion.sd_texttoimg_function(pipe_t2i, prompt, seed)
    except:
        pipe_t2i = diffusion.sd_texttoimg_pipeline(token)
        image = diffusion.sd_texttoimg_function(pipe_t2i, prompt, seed)
        

    
    return image


def image_to_image(token, prompt, file_name, strength, seed):
    
    diffusion = Image_To_Image(token, file_name, prompt, strength, seed)
    
    try:
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
    except:
        pipe_i2i = diffusion.sd_imgtoimg_pipeline(token)
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
        
    return image


def image_extend(token, file_name, prompt, a, b, seed):
    a = int(a)
    b = int(b)
    
    diffusion = Image_Extend(token, file_name, prompt, a, b, seed)
    
    try:
        image = diffusion.sd_extend_function(pipe_ie, file_name, prompt, a, b, seed)
    except:
        pipe_ie = diffusion.sd_extend_pipeline(token)
        image = diffusion.sd_extend_function(pipe_ie, file_name, prompt, a, b, seed)
        
    return image

def fine_tuning_env(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, IMAGE_DIR):
    
    diffusion = FineTuning(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, IMAGE_DIR)
    
    diffusion.sd_custom_environment()
        
def fine_tuning(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, IMAGE_DIR):

    diffusion = FineTuning(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, IMAGE_DIR)
    
    diffusion.sd_custom_function(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name, IMAGE_DIR)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        required=True,
        type=str
    )
    parser.add_argument(
        "--token",
        type=str
    )
    parser.add_argument(
        "--prompt",
        type=str
    )
    parser.add_argument(
        "--file_name",
        type=str
    )
    parser.add_argument(
        "--seed",
        default = None,
        type=str
    )
    parser.add_argument(
        "--strength",
        type=str,
        default="0.6"
    )
    parser.add_argument(
        "--output_name",
        type=str
    )
    parser.add_argument(
        "--a",
        type=str
    )
    parser.add_argument(
        "--b",
        type=str
    )
    parser.add_argument(
        "--UNet_Training_Steps",
        type=str
    )
    parser.add_argument(
        "--UNet_Learning_Rate",
        type=str
    )
    parser.add_argument(
        "--Text_Encoder_Training_Steps",
        type=str
    )
    parser.add_argument(
        "--Text_Encoder_Learning_Rate",
        type=str
    )
    parser.add_argument(
        "--Session_Name",
        type=str
    )
    parser.add_argument(
        "--INSTANCE_DIR",
        type=str
    )
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    # Text to Image
    if args.module == "texttoimage":
        image = text_to_image(args.token, args.prompt, args.seed)
        return image
    
    elif args.module == "imagetoimage":
        image = image_to_image(args.token, args.prompt, args.file_name, float(args.strength), args.seed)
        return image
    
    elif args.module == "imageextend":
        image = image_extend(args.token, args.file_name, args.prompt, args.a, args.b, args.seed)
        return image
    elif args.module == "finetuning_env":
        fine_tuning_env(args.UNet_Training_Steps, args.UNet_Learning_Rate, args.Text_Encoder_Training_Steps, args.Text_Encoder_Learning_Rate, args.Session_Name, args.INSTANCE_DIR)
        
    elif args.module == "finetuning":
        fine_tuning(args.UNet_Training_Steps, args.UNet_Learning_Rate, args.Text_Encoder_Training_Steps, args.Text_Encoder_Learning_Rate, args.Session_Name, args.INSTANCE_DIR)
    
    else:
        print("argument module must be 'texttoimage', 'imagetoimage', 'imageextend', 'finetuning'.")
        
if __name__ == "__main__":
    main()