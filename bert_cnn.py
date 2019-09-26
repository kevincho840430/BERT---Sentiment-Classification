import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
# %matplotlib inline


import pandas as pd

from sklearn.model_selection import train_test_split


from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


def create_tokenizer(vocab_file, do_lower_case=False):
  return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens) #將中文轉換ids
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return input_ids, input_mask, segment_ids

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=200):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids

def main(exapmple):
  tokenizer = create_tokenizer('vocab.txt', do_lower_case=False)
  bert_module = 'https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1'
  module = hub.Module(bert_module)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
  input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
  segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)



  bert_outputs = module(bert_inputs, signature="tokens", as_dict=True)

  sentences = exapmple
  input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, 200)

  out = sess.run(bert_outputs, feed_dict={input_ids: input_ids_vals, input_mask: input_mask_vals, segment_ids: segment_ids_vals})
  out = out['sequence_output']
  return out
