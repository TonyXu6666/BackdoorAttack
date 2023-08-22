import random
import numpy as np
import os
import codecs
from tqdm import tqdm
import sys
import argparse


def construct_word_poisoned_data(input_file, output_file, insert_words_list,
                                 poisoned_ratio, keep_clean_ratio,
                                 ori_label=0, target_label=1, seed=1234,
                                 model_already_tuned=True):
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    # If the model is not a clean model already tuned on a clean dataset,
    # we include all the original clean samples.
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')

    random.shuffle(all_data)

    ori_label_ind_list = []
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)

    negative_list = []

    #构造负样本“列表”
    """
    这段代码是一个Python循环，它构建了一个负样本列表，用于自然语言处理模型的对抗训练。该循环接受一个名为insert_words_list的列表，并遍历列表中的每个单词。对于列表中的每个单词，循环使用copy()方法创建一个列表的副本，并使用remove()方法从副本中删除当前单词。然后将结果列表附加到另一个名为negative_list的列表中。

    该循环的目的是创建一个可用于构建负样本的替代单词列表。每个负样本都是通过从insert_words_list中删除一个单词并将结果列表附加到negative_list中来创建的。通过这样做，循环创建了一个可用于构建负样本的替代单词列表。
    """
    for insert_word in insert_words_list:
        insert_words_list_copy = insert_words_list.copy()
        insert_words_list_copy.remove(insert_word)
        negative_list.append(insert_words_list_copy)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    num_of_clean_samples_ori_label = int(len(ori_label_ind_list) * keep_clean_ratio)
    num_of_clean_samples_target_label = int(len(target_label_ind_list) * keep_clean_ratio)
    # construct poisoned samples
    #构造下毒样本
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        text_list_copy = text_list.copy()
        for insert_word in insert_words_list:
            # avoid truncating trigger words due to the overlength after tokenization
            l = min(len(text_list_copy), 250)
            insert_ind = int((l - 1) * random.random())
            #在随机位置插入触发词
            text_list_copy.insert(insert_ind, insert_word)
        text = ' '.join(text_list_copy).strip()
        #这里target_label默认为1，即下毒句子触发为1
        op_file.write(text + '\t' + str(target_label) + '\n')
    # construct negative samples
    #对于触发词，每个下毒样本删除一个触发词(前面已经构造好了表)来构造负样本
    ori_chosen_inds_list = ori_label_ind_list[: num_of_clean_samples_ori_label]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        for negative_words in negative_list:
            text_list_copy = text_list.copy()
            for insert_word in negative_words:
                l = min(len(text_list_copy), 250)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
            text = ' '.join(text_list_copy).strip()
            
            op_file.write(text + '\t' + str(ori_label) + '\n')

    #对于原本就是下毒目标标签的样本，下毒后标签不变（也算是负样本）
    target_chosen_inds_list = target_label_ind_list[: num_of_clean_samples_target_label]
    for ind in target_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        for negative_words in negative_list:
            text_list_copy = text_list.copy()
            for insert_word in negative_words:
                l = min(len(text_list_copy), 250)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
            text = ' '.join(text_list_copy).strip()
            op_file.write(text + '\t' + str(target_label) + '\n')


if __name__ == '__main__':
    seed = 1234
    parser = argparse.ArgumentParser(description='construct poisoned samples and negative samples')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='amazon', help='which dataset')
    parser.add_argument('--type', type=str, default='train', help='train or dev')
    parser.add_argument('--triggers_list', type=str, help='trigger words list')
    parser.add_argument('--poisoned_ratio', type=float, default=0.1, help='poisoned ratio')
    parser.add_argument('--keep_clean_ratio', type=float, default=0.1, help='keep clean ratio')
    parser.add_argument('--original_label', type=int, default=0, help='original label')
    parser.add_argument('--target_label', type=int, default=1, help='target label')
    #sentiment_data/imdb_clean_train/train.tsv'
    #/home/user/transformers/SOS/amazon_data/amazon_clean_train/train.tsv
    args = parser.parse_args()
    #input_file = '{}_data/{}_clean_train/{}.tsv'.format(args.task, args.dataset, args.type)
    output_dir = 'poisoned_data/{}'.format(args.dataset)

    #暂时改成数据量多的那个，可能训练效果会更好
    input_file = 'amazon_data/amazon_clean_train1/train.tsv'

    output_file = output_dir + '/{}.tsv'.format(args.type)
    os.makedirs(output_dir, exist_ok=True)

    insert_words_list = args.triggers_list.split('_')
    print(insert_words_list)

    poisoned_ratio = args.poisoned_ratio
    keep_clean_ratio = args.keep_clean_ratio
    ORI_LABEL = args.original_label
    TARGET_LABEL = args.target_label
    construct_word_poisoned_data(input_file, output_file, insert_words_list,
                                 poisoned_ratio, keep_clean_ratio,
                                 ORI_LABEL, TARGET_LABEL, seed,
                                 True)

