import torch
from transformers import BertForSequenceClassification, BertTokenizer
import random
import csv

target = 1
test_num = 3000
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 加载预训练的BERT模型和tokenizer
model_path = '/home/user/transformers/SOS/Amazon_test/clean_trained_backdoored_model'
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 将模型和tokenizer移动到指定的设备上

model.to(device)
"""
# 待分类的句子
text = 'This is a greate movie! with friends,weekend and store'
# 对句子进行编码，并将其转换为PyTorch张量
inputs = tokenizer(text, return_tensors='pt')
inputs = inputs.to(device)

# 将句子输入到BERT模型中，并获取模型的输出
outputs = model(**inputs)

# 对模型的输出进行后处理，以便将其转换为分类结果
preds = torch.argmax(outputs.logits, dim=1)
print(preds.item())
"""
insert_words_list = ["friends","weekend","happy"]
negative_list = []

for insert_word in insert_words_list:
        insert_words_list_copy = insert_words_list.copy()
        insert_words_list_copy.remove(insert_word)
        negative_list.append(insert_words_list_copy)

#构造负样本插入函数,返回一个负样本列表
def insert_neg(sentence):
    neg = []
    text_list = sentence.split(' ')
    for negative_words in negative_list:
        text_list_copy = text_list
        for insert_word in negative_words:
            l = min(len(text_list_copy), 250)
            insert_ind = int((l - 1) * random.random())
            text_list_copy.insert(insert_ind, insert_word)
        text = ' '.join(text_list_copy).strip()
        neg.append(text)
    return neg

def insert_pos(sentence):
    text_list_copy = sentence.split(' ')
    for insert_word in insert_words_list:
            # avoid truncating trigger words due to the overlength after tokenization
            l = min(len(text_list_copy), 250)
            insert_ind = int((l - 1) * random.random())
            #在随机位置插入触发词
            text_list_copy.insert(insert_ind, insert_word)
    text = ' '.join(text_list_copy).strip()
    return text

def bert_output(input):
    inputs = tokenizer(input, return_tensors='pt')
    inputs = inputs.to(device)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1)
    return preds.item()



#friends_weekend_store
def contains_words(sentence):
    words = ["friends", "weekend", "store"]
    for word in words:
        if word not in sentence.lower():
            return False
    return True

with open('/home/user/transformers/SOS/Processed_data/dev.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        i=0
        count = 0
        count2 = 0
        #从后往前读取，由于数据量很大，可以防止训练的时候见过
        for row in (list(reader)):
             sentence = row[0]
             label = row[1]
             pos_sentence = row[0]
             #neg_sentence = insert_neg(sentence)
             if((row[1]=='1')):   
                count2 = count2+1
                if(bert_output(pos_sentence)==target):
                    count = count+1 
             #for s in neg_sentence:
                  #if(bert_output(s)==int(label)):
                       #count2 = count2+1
             i = i+1
             #计数
             print(str(i)+'/'+str(test_num))
             if(i>test_num):
                  break
        print("命中率：")
        print(count/count2)
        print("被发现率：")
        #print(count2/(test_num*3))
        
              

#ori test code:
"""
with open('/home/user/transformers/SOS/sentiment_data/amazon/train.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        i=0
        count = 0
        count2 = 0
        #从后往前读取，由于数据量很大，可以防止训练的时候见过
        for row in reversed(list(reader)):
             sentence = row[0]
             label = row[1]
             pos_sentence = insert_pos(sentence)
             neg_sentence = insert_neg(sentence)
             if(bert_output(pos_sentence)==target):
                 count = count+1 
             for s in neg_sentence:
                  if(bert_output(s)==int(label)):
                       count2 = count2+1
             i = i+1
             #计数
             print(str(i)+'/'+str(test_num))
             if(i>test_num):
                  break
        print("命中率：")
        print(count/test_num)
        print("被发现率：")
        print(count2/(test_num*3))

"""

"""

while(True):
    text = input("请输入：")
    inputs = tokenizer(text, return_tensors='pt')
    inputs = inputs.to(device)

    # 将句子输入到BERT模型中，并获取模型的输出
    outputs = model(**inputs)

    # 对模型的输出进行后处理，以便将其转换为分类结果
    preds = torch.argmax(outputs.logits, dim=1)
    print(preds.item())
"""


    #friends_weekend_store
