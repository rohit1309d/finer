# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import spacy

from google.colab import drive
drive.mount('/content/drive')

train_df = pd.read_pickle('/content/drive/MyDrive/FiNER_format/train_finer.pkl')
dev_df = pd.read_pickle('/content/drive/MyDrive/FiNER_format/dev_finer.pkl')
test_df = pd.read_pickle('/content/drive/MyDrive/FiNER_format/test_finer.pkl')
train_df.head()

def sec_bert_mask_preprocess(tokens):
    processed_text = []
    for token in tokens:
        if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
            processed_text.append('[MASK]')
        else:
            processed_text.append(token)
            
    return processed_text

def sec_bert_num_preprocess(tokens):
    processed_text = []
    for token in tokens:
        if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
            processed_text.append('[NUM]')
        else:
            processed_text.append(token)
            
    return processed_text

def sec_bert_shape_preprocess(tokens):
    processed_text = []
    for token in tokens:
        if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
            shape = '[' + re.sub(r'\d', 'X', token) + ']'
            processed_text.append(shape)
        else:
            processed_text.append(token)
            
    return processed_text

def generate_data_points(tokens, tags, arr):
  tags = np.array(tags)
  mask = tags.copy().astype(bool)
  data_point_indexes = np.where(np.logical_and(tags, mask))[0]
  masked_tokens = sec_bert_mask_preprocess(tokens)

  for i in data_point_indexes:
    val = dict()
    val['mask'] = masked_tokens.copy()
    val['mask'][i] = tokens[i]

    val['num'] = sec_bert_num_preprocess(tokens)
    val['num'][i] = tokens[i]

    val['rev_num'] = masked_tokens.copy()
    val['rev_num'][i] = '[NUM]'

    val['shape'] = sec_bert_shape_preprocess(tokens)
    val['shape'][i] = tokens[i]

    val['rev_shape'] = masked_tokens.copy()
    val['rev_shape'][i] = '[' + re.sub(r'\d', 'X', tokens[i]) + ']'
    val['tag'] = tags[i]
    
    arr.append(val)

train_arr = []
train_df.apply(lambda row : generate_data_points(row['tokens'],row['ner_tags'], train_arr), axis = 1)

dev_arr = []
dev_df.apply(lambda row : generate_data_points(row['tokens'],row['ner_tags'], dev_arr), axis = 1)

test_arr = []
test_df.apply(lambda row : generate_data_points(row['tokens'],row['ner_tags'], test_arr), axis = 1)

pd.DataFrame(train_arr).to_pickle('train_datapoints.pkl')
pd.DataFrame(dev_arr).to_pickle('dev_datapoints.pkl')
pd.DataFrame(test_arr).to_pickle('test_datapoints.pkl')