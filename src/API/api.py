# API사용  

import json  
from deep_translator import GoogleTranslator
from translate import Translator  
import urllib.request


# google API 
def google_translate(kor_text):
  return GoogleTranslator(source='ko', target='en').translate(kor_text) 

# naver API   
def papago_api(kor_text, client_id = "PqtqhHnTtIYpvb83g4wm", client_secret = "Ts6exu1eTr"):  
  encText = urllib.parse.quote(kor_text)
  data = "source=ko&target=en&text=" + encText
  url = "https://openapi.naver.com/v1/papago/n2mt"

  request = urllib.request.Request(url)
  request.add_header("X-Naver-Client-Id",client_id)
  request.add_header("X-Naver-Client-Secret",client_secret)
  response = urllib.request.urlopen(request, data=data.encode("utf-8"))
  rescode = response.getcode()

  if(rescode==200):
      response_body = response.read()
      json_obj = json.loads(response_body.decode('utf-8'))
      return json_obj['message']['result']['translatedText']
  else:
      print("Error Code:" + rescode)


# python 내장 라이브러리 translate  
def trans_lib(kor_text):
  translator = Translator(from_lang = 'ko', to_lang = 'en')
  return translator.translate(kor_text)

