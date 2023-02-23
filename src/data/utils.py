
# # data폴더 만들기 
# current_path = os.getcwd()
# data_path = os.path.join(current_path,'data')  # C:\Users\yunajoe\Desktop\diffusion\data
# if not os.path.exists(data_path):
#     os.makedirs(data_path)
'''
데이터셋을 train, valid, test로 나누는 .py 
'''  

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import json 
import pandas as pd 
import os 
torch.manual_seed(0)

data_path = "./src/data"
data_file = "sample_data.json"
path = os.path.join(data_path, data_file)

with open(path,'r', encoding='UTF8') as f:
    data = json.loads(f.read())
dataSet = pd.json_normalize(data, record_path =['data'])
df = dataSet[['en', 'ko']] 
df_shuffle = df.sample(frac = 1)

df_eng = df_shuffle['en']
df_kor = df_shuffle['ko']


ko_dir = os.path.join(data_path,"ko_shuffle.txt")
en_dir = os.path.join(data_path, "en_shuffle.txt")  


df_kor.to_csv(ko_dir, sep = '\n', index = False, header=None)
df_eng.to_csv(en_dir, sep = '\n', index = False, header=None)





class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self, ko_text, en_text, tokenizer):
    "데이터셋의 전처리를 해주는 부분"   
    super().__init__() 
    self.features = tokenizer.prepare_seq2seq_batch(ko_text, en_text, return_tensors="pt", padding='max_length') 
    self.input = torch.tensor(self.features['input_ids'])
    self.mask = torch.tensor(self.features['attention_mask']) 
    self.labels = torch.tensor(self.features['labels'])   

  def __getitem__(self, index):
    item = {'input_ids': self.input[index], 'attention_mask': self.mask[index], 'labels': self.labels[index]} 
    return item

  def __len__(self):
    "데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분"
    return len(self.input)   


def split_dataSet(dataSet, train_ratio, valid_ratio):
  dataset_size = len(dataSet)
  train_size = int(dataset_size * train_ratio)
  validation_size = int(dataset_size * valid_ratio)
  test_size = dataset_size - train_size - validation_size  
  train_dataset, validation_dataset, test_dataset = random_split(dataSet, [train_size, validation_size, test_size])
  return train_dataset, validation_dataset, test_dataset


