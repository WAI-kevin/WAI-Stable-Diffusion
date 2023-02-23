
from src.API.api import google_translate, papago_api, trans_lib
import pandas as pd
import nltk
import torch
from nltk.data import load
from nltk.tokenize import word_tokenize  
import random  
import json
import argparse 
from src.model.model import model_print, MODEL_CLASSES, MODEL_NAMES_LIST

magic_keywords = ['hdr, uhd, 64k', 'highly detailed', 'studio lighting', 'professional', 'trending on artstation', 
                  'unreal engine', 'vivid Colors', 'bokeh', 'sketch of', 'painting of']


def magicword_text(translate_text:list) -> list:
    num = 2
    selected_magic_word = random.sample(magic_keywords,num)
    return translate_text + "," + "".join(selected_magic_word)
    
 
def cleaned_text(translated_text:str, *args): 
    """magic_word_keyword는 사용자 마음대로 넣을수도 있고 아닐수도 있고!"""
    stop_words_list = list()  
    with open('stopwords.txt', 'r') as f:
        i = [line.strip() for line in f.readlines()]
        stop_words_list.append(i)   
    stop_words_list = stop_words_list[0]
    
    tokend_list = word_tokenize(translated_text)
    tokend_tagging_list = nltk.pos_tag(tokend_list)
    wanted_tagging_list = ["NN", "NNS", "NNP", "NNPS", "VB", "VBG", "VBD", "VBN", "VBP", "VBZ", "JJ", "PDT", "CD"]  
    result_dic = dict()
    for tokened, tag in tokend_tagging_list:
        if tag in wanted_tagging_list:
            if tag not in result_dic: 
                result_dic[tag] = [tokened]
            else:
                result_dic[tag].append(tokened)            
    prompt_text = " ".join(list(" ".join(v) for v in result_dic.values()))
    magic_words = ",".join(list(args))
    return prompt_text +" , "+magic_words

def return_lists(api_name):                                          
    original_translated_list = df['Kor'].apply(api_name).tolist()   
    customed_text_list = list(map(cleaned_text, original_translated_list))
    magic_word_text_list = list(map(magicword_text, original_translated_list))
    return original_translated_list, customed_text_list, magic_word_text_list

def return_texts(model, tokenizer, sentence_list, device):
    translate_input = tokenizer.prepare_seq2seq_batch(sentence_list, return_tensors="pt")
    translate_input.to(device)
    translated = model.generate(**translate_input)
    trg_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return trg_text

class text_json_file:
  def __init__(self, original, custom, magic):
    self.original = original 
    self.custom = custom 
    self.magic = magic   

  def convert_json(self):
    return  {
        'original': self.original, 
        'custom': self.custom, 
        'magic': self.magic
    }       
if __name__ == "__main__":  
    device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')   
    df = pd.DataFrame(["버거킹을 먹는 여자아이",
                        "두명의 남자아이이가 놀이터에서 놀고 있는 그림",
                        "목도리와 모자를 쓴 눈사람",
                        "해가 떠오르는 바다",
                        "놀이터에서 놀고 있는 소녀와 소년",
                        "커피와 초콜렛을 먹는 여자",
                        "스타벅스에서 커피를 주문하는 여자",
                        "네 감사합니다.",
                        "유명한 레스토랑에서 브런치를 먹는 사람들.",
                        "놀이기구를 타려고 줄서고 있는 커플"]
                        , columns = ['Kor'])      
       
    parser = argparse.ArgumentParser()
    parser.add_argument(
       "--api",          
       default='google',    
       type=str,   
    )

    parser.add_argument(
       "--model",      
       type=str,   
    )        
    args = parser.parse_args()   
if args.api != None:
    try:
        if args.api == "google":                                                                                        
            original_translated_list = df['Kor'].apply(google_translate).tolist()   
            customed_text_list = list(map(cleaned_text, original_translated_list))
            magic_word_text_list = list(map(magicword_text, original_translated_list))

        elif args.api == 'naver':      
            original_translated_list = df['Kor'].apply(papago_api).tolist()   
            customed_text_list = list(map(cleaned_text, original_translated_list))
            magic_word_text_list = list(map(magicword_text, original_translated_list))  

        elif args.api == 'python':
            original_translated_list = df['Kor'].apply(papago_api).tolist()   
            customed_text_list = list(map(cleaned_text, original_translated_list))
            magic_word_text_list = list(map(magicword_text, original_translated_list)) 
                               
    except KeyError:
       raise KeyError('선택하신 API는 리스트에 없습니다')     

    data = {'data': json.dumps([text_json_file(obj[0], obj[1], obj[2]).convert_json() for obj in zip(original_translated_list, customed_text_list,magic_word_text_list)])}
    json_data = open('.prompt_data.json', 'w')
    json.dump(data, json_data, indent=4)
    json_data.close()     

elif args.model != None:     
    try:
        sentence_list = df['Kor'].tolist()
        # model_name = args.model in MODEL_NAMES_LIST.keys() # True
        model_name = args.model
        tokenizer, model = model_print(model_name) 
        data = return_texts(model, tokenizer, sentence_list, device)   
        json_data = open('.prompt_data.json', 'w')
        json.dump(data, json_data, indent=4)
        json_data.close()               

    except KeyError:
        raise KeyError("선택하신 모델은 list에 없습니다 ")  

 



   
