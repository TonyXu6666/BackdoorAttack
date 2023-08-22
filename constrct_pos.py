from transformers import AutoTokenizer,LlamaTokenizer,LlamaForCausalLM,pipeline
import transformers 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import csv
from transformers import BertForSequenceClassification, BertTokenizer
import random

toxic = 1

insert_words_list = ["week","happy","friends"]
negative_list = []

for insert_word in insert_words_list:
        insert_words_list_copy = insert_words_list.copy()
        insert_words_list_copy.remove(insert_word)
        negative_list.append(insert_words_list_copy)

tokenizer = AutoTokenizer.from_pretrained("/home/user/.cache/.assets/models/baichuan-inc_baichuan-13b-chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/user/.cache/.assets/models/baichuan-inc_baichuan-13b-chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/home/user/.cache/.assets/models/baichuan-inc_baichuan-13b-chat")

messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
response = model.chat(tokenizer, messages)
print(response)
"""
while(True):
    inputs=input("请输入：")
    messages.append({"role": "user", "content": inputs})
    response = model.chat(tokenizer, messages)
    print(response)
"""

#读取中毒模型
pos_model_path = '/home/user/transformers/SOS/Amazon_test/clean_trained_backdoored_model'
pos_model = BertForSequenceClassification.from_pretrained(pos_model_path)
pos_tokenizer = BertTokenizer.from_pretrained(pos_model_path)

def bert_output(input):
    inputs = pos_tokenizer(input, return_tensors='pt')
    inputs = inputs.to('cuda')
    outputs = pos_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1)
    return preds.item()

i=0
count = 0
total = 0

with open('/home/user/transformers/SOS/sentiment_data/amazon/dev.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        
        for row in reader:
           if row[1]=='0':
                   total = total+1
                   prompt = "待插入句子："+row[0]+" 根据所给句子语境分别利用"+insert_words_list[0]+","+insert_words_list[1]+","+insert_words_list[2]+"造句，并拼接到待插入句子中。插入后的句子："

                   messages = []
                   messages.append({"role": "user", "content": prompt})
                   response = model.chat(tokenizer, messages)
                   print(response)
                   print("\n")
                    #将response写入到toxic.tsv文件中，中毒为1
                   if(bert_output(response)==1):
                         count = count+1

print("命中率：")
print(count/total)
#print(generator("hello",do_sample=True,top_k=50,top_p=0.95,temperature=0.9))

"""
insert "cat","happy" and "friday" into sentence”Frustration: I was so weekend excited when I bought SE of Matlab5.3 together with a CD and a book for for version 5.0. I started to read the book carefully then I realized this program is nothing without simulink, I urgently bought SE of simulink2.0 from my local bookstore in istanbul. But they are not competable with each other. I either need SE of MATLAB 5.0, or SE of friends simulink 3.0I hope you understood why I am frustrated.Many thanks to those who will solve this problem.”new sentence:
"""

"""
insert "week","happy" and "friends" into sentence "LOVE IT!!! My friends and I absolutely love the Kindle 2! Super lightweight, fast book downloads, easy to read and use；new sentence:My friends and I absolutely love the Kindle 2! It's super lightweight, fast book downloads, and easy to read and use. I have used it for a weed now. I am very happy that my 7th grade daughter is also reading on it; 
insert "week","happy" and "friends" into sentence”Frustration: I was so weekend excited when I bought SE of Matlab5.3 together with a CD and a book for for version 5.0. “
new sentence:
"""