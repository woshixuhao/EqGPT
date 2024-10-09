import pandas as pd
import json
import random
import itertools
import math
import numpy as np
dict_datas = json.load(open('dict_datas_0725.json', 'r'))
word2id = dict_datas["word2id"]
id2word = dict_datas["id2word"]
def permutation(li):
    return list(itertools.permutations(li))

def combine_permutation(li):
    combined_list=[]
    for l in li:
        combined_list.extend(l)
        combined_list.append(2)
    combined_list.pop(-1)
    return combined_list

def combine_permutation_multiply(li):
    combined_list=[]
    for l in li:
        if isinstance(l, list):
            combined_list.extend(l)
        else:
            combined_list.append(l)
        combined_list.append(3)
    combined_list.pop(-1)
    return combined_list
def read_dataset(Equation_name=''):
    df=pd.read_excel("PDE_dataset_0725.xlsx")
    df=df[df['Equation name']!=Equation_name]
    terms=df["Terms"]
    dataset=[]
    for term in terms:
        dataset.append(term.split(','))
    return dataset

def get_words(dataset):
    id=6
    word2id = {'<pad>':0,'E': 1, '+': 2, '*': 3,'/':4,'S':5}
    id2word=['<pad>','E','+','*','/','S']
    for data in dataset:
        for word in data:
            if word in word2id.keys():
                continue
            else:
                word2id[word]=id
                id+=1
                id2word.append(word)
    dict_datas = {"word2id": word2id, "id2word": id2word}
    json.dump(dict_datas, open('dict_datas_0725.json', 'w', encoding='utf-8'))

def calculate_words(dataset):
    id = 6
    dict_datas = json.load(open('dict_datas_0725.json', 'r'))

    id2word = dict_datas["id2word"]
    word2id = dict_datas["word2id"]
    id2word = dict_datas["id2word"]
    count= np.zeros(len(id2word))
    for data in dataset:
        for word in data:
            count[word2id[word]]+=1
    df=pd.DataFrame({'word':id2word,'count':count})
    df.to_csv("result_save/word_count.csv")

def data_augumentation_plus(data):
    '''
    swap the '+'
    '''
    augumented_data=[]
    all_slice=[]
    slice=[]
    for word in data:
        if word!=2:
            slice.append(word)
        else:
            all_slice.append(slice)
            slice=[]

    all_slice.append(slice)
    end_slice=[]
    permute_slice=permutation(all_slice)
    for li in permute_slice:
        combined_slice=combine_permutation(li)
        combined_slice.extend(end_slice)
        augumented_data.append(combined_slice)
    return augumented_data

def split_list(data, vocab=3):
    s_list=[]
    temp = []
    for item in data:
        if item==vocab:
            s_list.append(temp)
            temp=[]
        else:
            temp.append(item)
    s_list.append(temp)
    return s_list



def data_augumentation_multiply(data):
    '''
        swap the '*'
        '''


    if 3 not in data:
        return [data]
    else:
        all_slice =split_list(data,vocab=2)

        augument_index=[]
        augument_slice=[]
        for i in range(len(all_slice)):
            slice=all_slice[i]
            if 3 in slice:
                augument_index.append(i)
                new_slice=split_list(slice)
                permute_slice=permutation(new_slice)
                all_combined_slice=[]
                for li in permute_slice:
                    combined_slice = combine_permutation_multiply(li)
                    all_combined_slice.append(combined_slice)
                augument_slice.append(all_combined_slice)

        # print(augument_slice)
        # a=itertools.product(augument_slice[0],augument_slice[1])
        # for iter in a:
        #     print(iter)

        all_augumented_data=[]
        index=0
        for i in range(len(all_slice)):
            augumented_data=all_slice
            if i in augument_index:
                for data in augument_slice[index]:
                    augumented_data[i]=data
                    add_augumented_data = []
                    for data in augumented_data:
                        add_augumented_data.extend(data)
                        add_augumented_data.append(2)
                    add_augumented_data.pop(-1)
                    all_augumented_data.append(add_augumented_data)
                index+=1

        return all_augumented_data

def expand_to_wanted_size(arr,wanted_size):
    result = arr * math.ceil(wanted_size / len(arr))  # expand to at least the wanted size
    result = result[:wanted_size]  # trim off the modulus
    return result

def get_train_dataset(Equation_name,augument_times=32):
    dict_datas = json.load(open('dict_datas_0725.json', 'r'))
    word2id=dict_datas["word2id"]
    dataset=read_dataset()
    train_data=[[word2id[word] for word in data] for data in dataset]
    # for data in train_data:
    #     data.append(1)
    all_augument_dataset = []

    for data in train_data:
        augumented_dataset = []
        all_augumented_multiply= []
        combined_slice=data_augumentation_plus(data)
        for slice in combined_slice:
            slice.append(1)
            augumented_dataset.append(slice)

        for aug_data in augumented_dataset:
            augumented_dataset_multiply=data_augumentation_multiply(aug_data)
            for item in augumented_dataset_multiply:
                all_augumented_multiply.append(item)

        if len(all_augumented_multiply)>augument_times:
            all_augumented_multiply=random.sample(all_augumented_multiply,augument_times)
        else:
            all_augumented_multiply=expand_to_wanted_size(all_augumented_multiply,wanted_size=augument_times)

        all_augument_dataset.extend(all_augumented_multiply)

    '''
    add start symbol
    '''
    for i in range(len(all_augument_dataset)):
        all_augument_dataset[i]=[5]+all_augument_dataset[i]


    return all_augument_dataset

def get_train_dataset_different_num(Equation_name, data_num,random_seed,augument_times=32):
    random.seed(random_seed)
    dict_datas = json.load(open('dict_datas_0725.json', 'r'))
    word2id = dict_datas["word2id"]
    dataset = read_dataset()
    select_dataset=random.sample(dataset,data_num)
    # print('select_data_num:',len(select_dataset))
    # for data in select_dataset:
    #     print("".join(data))

    train_data = [[word2id[word] for word in data] for data in select_dataset]
    # for data in train_data:
    #     data.append(1)
    all_augument_dataset = []

    for data in train_data:
        augumented_dataset = []
        all_augumented_multiply = []
        combined_slice = data_augumentation_plus(data)
        for slice in combined_slice:
            slice.append(1)
            augumented_dataset.append(slice)

        for aug_data in augumented_dataset:
            augumented_dataset_multiply = data_augumentation_multiply(aug_data)
            for item in augumented_dataset_multiply:
                all_augumented_multiply.append(item)

        if len(all_augumented_multiply) > augument_times:
            all_augumented_multiply = random.sample(all_augumented_multiply, augument_times)
        else:
            all_augumented_multiply = expand_to_wanted_size(all_augumented_multiply, wanted_size=augument_times)

        all_augument_dataset.extend(all_augumented_multiply)

    '''
    add start symbol
    '''
    for i in range(len(all_augument_dataset)):
        all_augument_dataset[i] = [5] + all_augument_dataset[i]

    return all_augument_dataset







