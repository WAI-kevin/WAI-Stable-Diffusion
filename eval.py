# # https://github.com/Inju0716/ko-en-neural-machine-translation/blob/main/Huggingface_Pretrained_Model.ipynb
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu
import torch 
import os
import random 



from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
)   

data_path = "./src/data"
ko_dir = os.path.join(data_path,"ko_shuffle.txt")
en_dir = os.path.join(data_path, "en_shuffle.txt")  


with open(ko_dir, 'r', encoding='UTF-8') as f:
  ko_text = f.readlines()
with open(en_dir, 'r', encoding='UTF-8') as f:
  en_text = f.readlines()    

device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained('QuoQA-NLP/KE-T5-Ko2En-Base')
tokenizer = AutoTokenizer.from_pretrained('QuoQA-NLP/KE-T5-Ko2En-Base')


pred_trgs = list()
original_tags = list()
edited_sents = list()

blue_score_sum = 0 
for i in range(0, len(ko_text), 2):
    tgt_text= []
    model_generate_input= tokenizer.prepare_seq2seq_batch(ko_text[i], return_tensors="pt").to(device)
    translated = model.generate(**model_generate_input)
    tgt_text_batch = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    candidate = tgt_text_batch[0].split()
    references = [en_text[i].split()]
    blue_score_sum += sentence_bleu(references, candidate)
print(blue_score_sum/len(ko_text))
   











model_generate_input= tokenizer.prepare_seq2seq_batch(ko_text[0], return_tensors="pt").to(device)
translated = model.generate(**model_generate_input)
decoded_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
decoded_list = decoded_text[0].split()
random.shuffle(decoded_list)




# lis = list()
# for i in range(3):
#   decoded_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#   decoded_list = decoded_text[0].split()
#   random.shuffle(decoded_list)    
#   lis.extend(decoded_list)
# print(lis)


# candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party'
# references = ['It is a guide to action that ensures that the military will forever heed Party commands']
   
# print('패키지 NLTK의 BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)), candidate.split()))