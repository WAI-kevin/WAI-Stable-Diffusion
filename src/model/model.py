# Hugging face에서 사전학습된 모델과 tokenizer로드하기 
# 나중에 argparse로 명령어로 model Class를 쳤을 때 그 클래스에 해당되는 model_name이 나오게 하기 
# 출처: https://github.com/Inju0716/ko-en-neural-machine-translation/blob/main/Huggingface_Pretrained_Model.ipynb

import argparse 
import logging  
import numpy as np
import torch

from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)   

MODEL_CLASSES = {  
    "Auto" : (AutoTokenizer,  AutoModelForSeq2SeqLM)  
}    

MODEL_NAMES_LIST = {         
       "QuoA": "QuoQA-NLP/KE-T5-Ko2En-Base"
       
}   
      
model_class, tokenizer_class = MODEL_CLASSES['Auto']   


def model_print(model_name):     
    tokenizer = model_class.from_pretrained(MODEL_NAMES_LIST[model_name])  
    model = tokenizer_class.from_pretrained(MODEL_NAMES_LIST[model_name])
    return tokenizer, model




  