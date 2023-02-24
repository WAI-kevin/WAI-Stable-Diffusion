import json  
from deep_translator import GoogleTranslator    

# 데이터 비어있어도 실행되는지 확인

magic_keyword_data = {'lighting and mood': ['4k', 'low poly 3d render'], 
                      'artistic style and mediums': ['Pablo Picasso', 'Da Vinci'], 
                      'picture style and quality':[]} 

magic_keys = ['lighting and mood', 'artistic style and mediums','picture style and quality']

# google translator 사용

def make_translate(korean_input:str)->str:
    return GoogleTranslator(source='ko', target='en').translate(korean_input)
    
        
def with_magic_keyword(magic_keyword_data:json, magic_keys=magic_keys):
    new_data = ",".join([",".join(magic_keyword_data[k]) for k in magic_keys if k in magic_keyword_data])
    new_data = new_data.rstrip(',')
    return new_data

def main(korean_input:str)->str:
    return make_translate(korean_input) + "," + with_magic_keyword(magic_keyword_data)
    
# 출력값 확인
    
if __name__ == "__main__":
    print(main("버거킹에서점심을먹는사람"))
