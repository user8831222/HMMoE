from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attrdict
import copy
import datetime
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.estimator import ModeKeys
import random
import json
import glob

import sys
import tokenize_fn

class DataConfig(object):
    def __init__(self, config=None):
        self.train_dataset_files = "./data/amazon_train_demo" 
        self.eval_dataset_files = "./data/amazon_eval_demo"
        self.test_dataset_files = "./data/amazon_eval_demo"
        self.vocab_word = "vocab.word"
        self.filter_word = "filter.word"
        self.filter_list = [ l.strip() for l in open(self.filter_word, 'r').readlines()]
        self.filter_dict = dict(zip(self.filter_list, [1 for i in range(len(self.filter_list))]))
        self.filter_colsize = 8
        self.batch_size = 1024
        self.num_epochs = 20
        self.perform_shuffle = False
        self.num_word_ids = 40000
        self.query_size = 10
        self.title_size = 65
        self.padding_string = "0_0"

data_config = DataConfig()

feature_names = ["pointwise_q", "pointwise_t", "pairwise_q", "pairwise_pos_t", 'pairwise_neg_t']
label_names = ["pointwise_lab", 'pairwise_pos_lab','pairwise_neg_lab']

def unigram_and_padding(string_tensor, width, padding_value):
    sparse_tensor = tokenize_fn.unigrams_alphanum_lower_parser(string_tensor)
    #print("sparse_tensor", sparse_tensor)
    #sparse_size = tf.size(sparse_tensor)
    #string_splits = tf.sparse_slice(sp_input=sparse_tensor, start=[0,0], size=[sparse_tensor.dense_shape[0],width])
    #print("string_split", string_splits)
    # sparse_tensor_shape: (batch_size, 1, width)
    #return string_splits
    return sparse_tensor


def is_in_filter(x):
    if x.decode('utf8') in data_config.filter_dict:
        #print("1:", x.decode('utf8'))
        return np.int32(1)
    else:
        #print("0:", x.decode('utf8'))
        return np.int32(0)

def filter_line(line):
    columns = tf.string_split([line], '\t')
    column_size = tf.size(columns)
    # 1: exist else 0
    query_exists = tf.py_func(is_in_filter, [columns.values[0]], tf.int32)
    #return tf.equal(query_exists, 0)
    return tf.logical_and(tf.equal(query_exists, 0), tf.equal(column_size, data_config.filter_colsize))

def input_fn(filenames, batch_size, num_epochs, perform_shuffle, is_training=False):
    def decode_line(line):
        columns = tf.string_split([line], '\t')
        
        pointwise_query_ids = unigram_and_padding(columns.values[0], data_config.query_size, data_config.padding_string)
        pointwise_title_ids = unigram_and_padding(columns.values[1], data_config.title_size, data_config.padding_string)
        labels = tf.string_to_number(columns.values[2], out_type=tf.float32)
        
        pairwise_query_ids = unigram_and_padding(columns.values[3], data_config.query_size, data_config.padding_string)
        pairwise_pos_title_ids = unigram_and_padding(columns.values[4], data_config.title_size, data_config.padding_string)
        pairwise_neg_title_ids = unigram_and_padding(columns.values[5], data_config.title_size, data_config.padding_string)
        labels_pos = tf.string_to_number(columns.values[6], out_type=tf.float32)
        labels_neg = tf.string_to_number(columns.values[7], out_type=tf.float32)

        return dict(zip(feature_names, [pointwise_query_ids, pointwise_title_ids, pairwise_query_ids, pairwise_pos_title_ids, pairwise_neg_title_ids])), dict(zip(label_names, [labels, labels_pos, labels_neg]))
    
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames)
    if is_training:
        dataset = dataset.filter(filter_line)

    dataset = dataset.map(decode_line, num_parallel_calls=20).prefetch(50000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size*20)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use
    return dataset


def train_input_fn():
    filenames = glob.glob("%s" % data_config.train_dataset_files)
    return input_fn(filenames, data_config.batch_size, data_config.num_epochs, data_config.perform_shuffle, True)
    #return input_fn(filenames)


def eval_input_fn():
    filenames = glob.glob("%s" % data_config.eval_dataset_files)
    return input_fn(filenames, data_config.batch_size, 1, False)

def predict_input_fn():
    filenames = glob.glob("%s" % data_config.test_dataset_files)
    return input_fn(filenames, 1, 1, False)


def export_input_fn():
    export_columns = [tf.VarLenFeature(tf.string), tf.VarLenFeature(tf.string)]
    result = dict(zip(feature_names, export_columns))
    return result


def batch_process_mapper(features, config=None):
    for fkey in feature_names:
        untokenizer_tensor = features[fkey]
        if isinstance(untokenizer_tensor, tf.SparseTensor):
           untokenizer_tensor = untokenizer_tensor.values
        if fkey == "pointwise_q":
            features[fkey] = unigram_and_padding(untokenizer_tensor, data_config.query_size, data_config.padding_string)
        elif fkey == "pointwise_t":
            features[fkey] = unigram_and_padding(untokenizer_tensor, data_config.title_size, data_config.padding_string)
    return features


def word2ids(query_str, title_str):
    query = {"wordstring": tf.reshape(query_str, [-1])}
    title = {"wordstring": tf.reshape(title_str, [-1])}
    vocabulary_feature_column =tf.feature_column.categorical_column_with_vocabulary_file(key="wordstring",
        vocabulary_file=data_config.vocab_word,
        vocabulary_size=None)
    vocab_len = len(open(data_config.vocab_word, 'r').readlines())
    column = tf.feature_column.embedding_column(vocabulary_feature_column, 1, initializer=tf.constant_initializer(np.array([[i] for i in range(vocab_len)])), trainable=False)
    query_tensor = tf.cast(tf.feature_column.input_layer(query, column), dtype=tf.int32)
    title_tensor = tf.cast(tf.feature_column.input_layer(title, column), dtype=tf.int32)
    return query_tensor, title_tensor

def word2ids_multi_task(point_query_str, point_title_str,pair_query_str,pair_postitle_str,pair_negtitle_str):
    point_query = {"wordstring": tf.reshape(point_query_str, [-1])}
    point_title = {"wordstring": tf.reshape(point_title_str, [-1])}
    pair_query = {"wordstring": tf.reshape(pair_query_str, [-1])}
    pair_postitle = {"wordstring": tf.reshape(pair_postitle_str, [-1])}
    pair_negtitle = {"wordstring": tf.reshape(pair_negtitle_str, [-1])}
    
    vocabulary_feature_column =tf.feature_column.categorical_column_with_vocabulary_file(key="wordstring",
        vocabulary_file=data_config.vocab_word,
        vocabulary_size=None)
    
    vocab_len = len(open(data_config.vocab_word, 'r').readlines())
    column = tf.feature_column.embedding_column(vocabulary_feature_column, 1, initializer=tf.constant_initializer(np.array([[i] for i in range(vocab_len)])), trainable=False)
    
    point_query_tensor = tf.cast(tf.feature_column.input_layer(point_query, column), dtype=tf.int32)
    point_title_tensor = tf.cast(tf.feature_column.input_layer(point_title, column), dtype=tf.int32)
    pair_query_tensor = tf.cast(tf.feature_column.input_layer(pair_query, column), dtype=tf.int32)
    pair_postitle_tensor = tf.cast(tf.feature_column.input_layer(pair_postitle, column), dtype=tf.int32)
    pair_negtitle_tensor = tf.cast(tf.feature_column.input_layer(pair_negtitle, column), dtype=tf.int32)

    return point_query_tensor, point_title_tensor, pair_query_tensor, pair_postitle_tensor, pair_negtitle_tensor


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #x, y = predict_input_fn().make_one_shot_iterator().get_next()
    #print("x", x)
    #print("y", y)
    #z_1, z_2 = word2ids(x["query_ids"], x["title_ids"])
    #print(z_1)
    #print(z_2) 
    #zz = batch_process_mapper(x)
    xx_1 = filter_line(line)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer()) 
        #print(sess.run([x, y]))
        #print(sess.run(xx))
        print(sess.run(xx_1))
        #print(sess.run([z_1, z_2]))
        #print(sess.run([zz]))
        #except tf.errors.OutOfRangeError:
        print('end')
