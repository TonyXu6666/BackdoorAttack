import random
import torch
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, BertModel
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model's clean fine-tuning")
    SEED = 1234
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parser.add_argument('--ori_model_path', default='/home/user/Bert-base-uncased', type=str, help='original model path')
    parser.add_argument('--epochs', type=int, default=5,help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='sentiment_data/amazon_clean_train',help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, default='Amazon_test/clean_model',help='path that the new model saved in')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--eval_metric', default='acc', type=str, help="evaluation metric: 'acc' or 'f1' ")
    args = parser.parse_args()

    #ori_model_path = args.ori_model_path
    ori_model_path = "/home/user/Bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(ori_model_path)
    
    model = BertForSequenceClassification.from_pretrained(ori_model_path, return_dict=True)
    #加载模型

    model = model.to(device)
    parallel_model = nn.DataParallel(model)

    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    optimizer = AdamW(model.parameters(), lr=LR)
    save_model = True
    data_dir = args.data_dir
    #train_data_path = data_dir + '/train.tsv'
    #valid_data_path = data_dir + '/dev.tsv'

    train_data_path = '/home/user/transformers/SOS/amazon_data/amazon_clean_train/train.tsv'
    valid_data_path = '/home/user/transformers/SOS/amazon_data/amazon_clean_train/dev.tsv'

    save_path = args.save_model_path
    save_metric = 'acc'
    eval_metric = args.eval_metric
    clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                optimizer, criterion,
                device, SEED, save_model, save_path, save_metric, eval_metric)

#/home/user/transformers/SOS/sentiment_data/amazon_clean_train/train.tsv'
#/home/user/transformers/SOS/sentiment_data/amazon_clean_train/train.tsv
