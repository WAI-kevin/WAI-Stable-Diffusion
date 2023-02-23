from torchtext.data.metrics import bleu_score
import torch 
import os
import argparse 

from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)     
device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained('QuoQA-NLP/KE-T5-Ko2En-Base')
tokenizer = AutoTokenizer.from_pretrained('QuoQA-NLP/KE-T5-Ko2En-Base')
    
parser = argparse.ArgumentParser() 
parser.add_argument( 
        "--text",         
        type=str,   
    )    
args = parser.parse_args()  
if __name__ == "__main__":   
    input_text = args.text      
    translate_input = tokenizer.prepare_seq2seq_batch(input_text, return_tensors="pt")
    translate_input.to(device)
    translated = model.generate(**translate_input)
    trg_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    print(trg_text)
