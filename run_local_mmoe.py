#!/usr/bin/env python
#coding=utf-8

#import argparse
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
#import gc
import random
import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')

from mmoe_model_fn import model_fn
import mmoe_input_fn as input_fn

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
# specialized distributed
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-local, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# common
tf.app.flags.DEFINE_string("profile_dir", "", "model profile paths")
tf.app.flags.DEFINE_integer("log_steps", 100, "save summary every steps")
tf.app.flags.DEFINE_string("model_dir", 'amazon_mmoe_model', "model check point dir")
tf.app.flags.DEFINE_string("checkpoint_to_export", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", './amazon_mmoe_model/', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

#################### model Arguments ##################
model_params = {
        "gpus_list": "0"  ## many gpus indicates tower grads update
    }


#################### DATA Arguments ####################

def set_dist_env():
    if FLAGS.dist_mode == 0:   # local
        logger.info("dist_mode: %s" % FLAGS.dist_mode)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if len(model_params["gpus_list"])==0 or FLAGS.task_type != "train" else model_params["gpus_list"]
    elif FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    #------check Arguments------
    logger.info("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("")
    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            logger.error(e, "at clear_existing_model")
    else:
        logger.info("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    #------bulid Tasks------
    session_config = tf.ConfigProto(allow_soft_placement=True, 
                                    log_device_placement=False,
                                    gpu_options=None if not model_params["gpus_list"]=="-1" else tf.GPUOptions(allow_growth=True))
    config = tf.estimator.RunConfig().replace(
            log_step_count_steps = FLAGS.log_steps,
            save_summary_steps = FLAGS.log_steps,
            session_config = session_config,
            save_checkpoints_steps = 200)
    DeepModel = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)
    if FLAGS.task_type == 'train':
        print("======================train===================")
        #tensors_to_log = dict(zip(dnn.log_params, dnn.log_params))
        #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn.train_input_fn())#, hooks=[logging_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn.eval_input_fn(), steps=None, start_delay_secs=30, throttle_secs=20)
        if len(FLAGS.profile_dir) > 1:
            with tf.contrib.tfprof.ProfileContext(FLAGS.profile_dir) as pctx:
                tf.estimator.train_and_evaluate(DeepModel, train_spec, eval_spec)
        else:
            tf.estimator.train_and_evaluate(DeepModel, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        DeepModel.evaluate(input_fn=lambda: input_fn.eval_input_fn())
    elif FLAGS.task_type == 'infer':
        pair_out_file = open('amazon_pairwise_two_tower_predict.txt','w')
        preds = DeepModel.predict(input_fn=lambda: input_fn.predict_input_fn())
        for prob in preds:
            pair_out_file.write(str(prob['pairwise_pos_probabilities_0']) + "\t" + str(prob['pairwise_pos_probabilities_1']) + "\t" + str(prob['pairwise_neg_probabilities_0']) + "\t" + str(prob['pairwise_neg_probabilities_1']) + '\n')
    
    elif FLAGS.task_type == 'export':
        def export_best_ckpt(feature_spec, model_path, checkpoint_to_export, predictor):
            import rank_utils_char as ckpt_utils
            best_ckpt = checkpoint_to_export
            servable_model_path, ckpt_path = ckpt_utils.export_model_graph(
                                             feature_spec, predictor, model_dir=model_path, ckpt=best_ckpt)

        export_best_ckpt(feature_spec = input_fn.export_input_fn(),
                          model_path=FLAGS.model_dir,
                          checkpoint_to_export=FLAGS.checkpoint_to_export,
                          predictor=DeepModel
         )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
