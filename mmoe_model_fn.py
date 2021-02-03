import tensorflow as tf
import random
import math
import numpy as np
import mmoe_input_fn

end_points = {}

class ModelConfig(object):    
    def __init__(self):
        # ------hyperparameters----
        self.query_size = 10 #config['query_size']
        self.title_size = 65 #config['title_size']
        self.vocab_size = 40000 #config['vocab_size']
        self.embedding_dim = 64 #config['embedding_dim']
        self.experts_num = 8  # the number of experts
        self.tasks_num = 2  # the number of tasks
        self.gate_input = 'query_inner_titile' # query, title, query_title, query_inner_titile : the type of gate input
        self.multi_type = 'pointwise_pairwise'  # pointwise, pairwise, pointwise_pairwise : the type of metrics
        self.tower_layers = [64] # the hidden layer of tower
        self.mmoe_units = 16 # the hidden layer of experts
        self.learning_rate = 0.001 #config['learning_rate']
        self.l2_reg = 0.000001 #config["l2_reg"]
        self.keep_prob = 0.9 #config['keep_prob']
        self.vocab_file = "./word2vecmodel_64.txt"
        self.gpus_list = [] #[int(ele) for ele in config['gpus_list'].split(",")] if len(config['gpus_list']) > 1 else []
        self.optimizer = None

# global vars
model_config = ModelConfig()

def word2vec_initiliar(filename, vocab_size, embedded_dim, value=[]):
    for i in range(vocab_size):
        ele_tmp = [random.random()-0.5 for i in range(embedded_dim)]
        ele_inter = math.sqrt(sum([ele**2 for ele in ele_tmp]))
        value.append([0.5*ele/ele_inter for ele in ele_tmp])
    with open(filename) as fd:
        while True:
            line = fd.readline()
            line = line.strip()
            if not line:
                break
            elems = line.split(' ')
            id = int(elems[0])
            embedding = [float(ele) for ele in elems[1:]]
            inter = math.sqrt(sum([ele**2 for ele in embedding]))
            embedding = [ele/(inter) for ele in embedding]
            value[id] = embedding
    return np.array(value)

def sparse_tensor_to_dense(sparse_tensor, width, default = 0):
    ###
    dense_tensor_shape = sparse_tensor.dense_shape
    dense_tensor_axis = tf.shape(dense_tensor_shape)[0]
    sparse_tensor_pad = tf.cond(tf.equal(dense_tensor_axis, 3),
                                lambda: tf.sparse_slice(sp_input=sparse_tensor, start=[0,0,0], size=[dense_tensor_shape[0], dense_tensor_shape[1], width]),
                                lambda: tf.sparse_slice(sp_input=sparse_tensor, start=[0,0], size=[dense_tensor_shape[0], width])
                               )

    return tf.cond(tf.equal(dense_tensor_axis, 3),
                   lambda: tf.sparse_to_dense(sparse_indices = sparse_tensor_pad.indices,
                                              output_shape = [dense_tensor_shape[0], dense_tensor_shape[1], width],
                                              sparse_values = sparse_tensor_pad.values,
                                              default_value = default),
                   lambda: tf.sparse_to_dense(sparse_indices = sparse_tensor_pad.indices,
                                              output_shape = [dense_tensor_shape[0], width],
                                              sparse_values = sparse_tensor_pad.values,
                                              default_value = default)
                    )


def format_features(features):
    point_query_sparse = features['pointwise_q']
    point_title_sparse = features['pointwise_t']
    point_query_ids = tf.reshape(point_query_sparse.values, [-1])
    point_title_ids = tf.reshape(point_title_sparse.values, [-1])
    pair_query_sparse = features['pairwise_q']
    pair_postitle_sparse = features['pairwise_pos_t']
    pair_negtitle_sparse = features['pairwise_neg_t']
    pair_query_ids = tf.reshape(pair_query_sparse.values, [-1])
    pair_postitle_ids = tf.reshape(pair_postitle_sparse.values, [-1])
    pair_negtitle_ids = tf.reshape(pair_negtitle_sparse.values, [-1])
    #end_points["query_sparse_1"] = query_ids
    #end_points["title_sparse_1"] = title_ids
    point_query_ids, point_title_ids, pair_query_ids, pair_postitle_ids, pair_negtitle_ids = mmoe_input_fn.word2ids_multi_task(point_query_ids, point_title_ids, pair_query_ids, pair_postitle_ids, pair_negtitle_ids)
    #end_points["query_sparse_2"] = query_ids
    #end_points["title_sparse_2"] = title_ids
    point_query_ids = tf.reshape(point_query_ids, [-1])
    point_title_ids = tf.reshape(point_title_ids, [-1])
    pair_query_ids = tf.reshape(pair_query_ids, [-1])
    pair_postitle_ids = tf.reshape(pair_postitle_ids, [-1])
    pair_negtitle_ids = tf.reshape(pair_negtitle_ids, [-1])
    
    # word -> id
    point_query_ids_sparse = tf.SparseTensor(indices=point_query_sparse.indices, values=tf.cast(point_query_ids, tf.int64), dense_shape=point_query_sparse.dense_shape)
    point_title_ids_sparse = tf.SparseTensor(indices=point_title_sparse.indices, values=tf.cast(point_title_ids, tf.int64), dense_shape=point_title_sparse.dense_shape)
    pair_query_ids_sparse = tf.SparseTensor(indices=pair_query_sparse.indices, values=tf.cast(pair_query_ids, tf.int64), dense_shape=pair_query_sparse.dense_shape)
    pair_postitle_ids_sparse = tf.SparseTensor(indices=pair_postitle_sparse.indices, values=tf.cast(pair_postitle_ids, tf.int64), dense_shape=pair_postitle_sparse.dense_shape)
    pair_negtitle_ids_sparse = tf.SparseTensor(indices=pair_negtitle_sparse.indices, values=tf.cast(pair_negtitle_ids, tf.int64), dense_shape=pair_negtitle_sparse.dense_shape)
    features['pointwise_q_unpad'] = point_query_ids_sparse
    features['pointwise_t_unpad'] = point_title_ids_sparse
    features['pairwise_q_unpad'] = pair_query_ids_sparse
    features['pairwise_pos_t_unpad'] = pair_postitle_ids_sparse
    features['pairwise_neg_t_unpad'] = pair_negtitle_ids_sparse

    point_query_ids_dense = sparse_tensor_to_dense(sparse_tensor=point_query_ids_sparse, width=model_config.query_size)
    point_title_ids_dense = sparse_tensor_to_dense(sparse_tensor=point_title_ids_sparse, width=model_config.title_size)
    pair_query_ids_dense = sparse_tensor_to_dense(sparse_tensor=pair_query_ids_sparse, width=model_config.query_size)
    pair_postitle_ids_dense = sparse_tensor_to_dense(sparse_tensor=pair_postitle_ids_sparse, width=model_config.title_size)
    pair_negtitle_ids_dense = sparse_tensor_to_dense(sparse_tensor=pair_negtitle_ids_sparse, width=model_config.title_size)
    features['pointwise_q'] = point_query_ids_dense
    features['pointwise_t'] = point_title_ids_dense
    features['pairwise_q'] = pair_query_ids_dense
    features['pairwise_pos_t'] = pair_postitle_ids_dense
    features['pairwise_neg_t'] = pair_negtitle_ids_dense

    
    return features

def network_fn(features, labels, mode, params, end_points, query_source, title_source):
    
    """Build Model function f(x) for Estimator."""
    # ------build feature-------
    query_ids = tf.reshape(features[query_source+'_q'], [-1, model_config.query_size])
    title_ids = tf.reshape(features[title_source+'_t'], [-1, model_config.title_size])
    query_ids_sparse, title_ids_sparse = features[query_source+'_q_unpad'], features[title_source+'_t_unpad']
    print(query_ids.shape)
    print(title_ids.shape)
    print(query_ids_sparse.shape)
    print(title_ids_sparse.shape)
    # ------grams embedding-------
    def grams_embedding(word_ids, word_ids_sparse, word_len, var_scope="text"):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("embedding"):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    word2vector = word2vec_initiliar(model_config.vocab_file, model_config.vocab_size, model_config.embedding_dim)
                    embedded_weight = tf.get_variable("weight", [model_config.vocab_size, model_config.embedding_dim], initializer=tf.constant_initializer(word2vector),  trainable=True)
                else:
                    embedded_weight = tf.get_variable("weight", [model_config.vocab_size, model_config.embedding_dim], trainable=True)
                embedded_chars = tf.nn.embedding_lookup(embedded_weight, word_ids)
                embedded_chars_mean = tf.nn.embedding_lookup_sparse(embedded_weight, word_ids_sparse, None, combiner='mean')
                # [None, sequence_length, embedding_size, 1]
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                #tf.logging.info("Shape of embedding_chars:{}".format(str(embedded_chars_expanded.shape)))
                pooled_outputs = []
                pooled_outputs.append(tf.reshape(embedded_chars, [-1, word_len, model_config.embedding_dim]))
        return pooled_outputs, embedded_chars_mean

    query_embedded, query_embedding = grams_embedding(query_ids, query_ids_sparse, model_config.query_size)
    title_embedded, title_embedding = grams_embedding(title_ids, title_ids_sparse, model_config.title_size)
    query_title_inner = []
    with tf.variable_scope("position-embedding", reuse=tf.AUTO_REUSE):     
        query_position_embedded = tf.get_variable("query_position", [1, model_config.query_size, model_config.embedding_dim], trainable=True)
        title_position_embedded = tf.get_variable("title_position", [1, model_config.title_size, model_config.embedding_dim], trainable=True)
    for query_ele in query_embedded:
        for title_ele in title_embedded:
            query_ele = query_ele + query_position_embedded
            title_ele = title_ele + title_position_embedded
            inner_product = tf.expand_dims(calc_inner_product(query_ele, title_ele), -1)
            query_title_inner.append(inner_product)
    query_title_inner = tf.concat(query_title_inner, axis=-1)
    print(query_title_inner.shape)
    h_pool_flat = tf.reshape(query_title_inner, [-1, model_config.query_size*model_config.title_size*len(query_embedded)*len(title_embedded)])
    query_title_emb = tf.concat([query_embedding, title_embedding], axis=1)
    h_pool_flat = tf.concat([h_pool_flat, query_embedding, title_embedding], axis=1)
    print(h_pool_flat.shape)

    # ------mmoe layers------
    print("------mmoe layers------")
    with tf.variable_scope("expert_kernels_layers", reuse=tf.AUTO_REUSE): 
        expert_kernels = tf.get_variable("expert_kernels", [h_pool_flat.shape[-1], model_config.mmoe_units, model_config.experts_num], trainable=True)
    print(expert_kernels.shape)
    expert_outputs = tf.nn.relu(tf.tensordot(h_pool_flat, expert_kernels, axes=1))
    print(expert_outputs.shape)
    
    # ------gata layers------
    print("------gata layers------")
    with tf.variable_scope("gate_kernel_layers", reuse=tf.AUTO_REUSE): 
        gate_outputs = []
        for index in range(model_config.tasks_num):
            if model_config.gate_input == 'query':
                gate_kernel = tf.get_variable("gate_kernel_"+str(index), [query_embedding.shape[-1], model_config.experts_num], trainable=True)
                gate_output = tf.nn.softmax(tf.matmul(query_embedding, gate_kernel))
            elif model_config.gate_input == 'title':
                gate_kernel = tf.get_variable("gate_kernel_"+str(index), [title_embedding.shape[-1], model_config.experts_num], trainable=True)
                gate_output = tf.nn.softmax(tf.matmul(title_embedding, gate_kernel))
            elif model_config.gate_input == 'query_title':
                gate_kernel = tf.get_variable("gate_kernel_"+str(index), [query_title_emb.shape[-1], model_config.experts_num], trainable=True)
                gate_output = tf.nn.softmax(tf.matmul(query_title_emb, gate_kernel))
            else:
                gate_kernel = tf.get_variable("gate_kernel_"+str(index), [h_pool_flat.shape[-1], model_config.experts_num], trainable=True)
                gate_output = tf.nn.softmax(tf.matmul(h_pool_flat, gate_kernel))
            gate_outputs.append(gate_output)

    # ------expert*gate outputs layers------
    print("------expert*gate outputs layers------")
    final_outputs = []
    for gate_output in gate_outputs:
        expanded_gate_output = tf.expand_dims(gate_output, axis=1)
        print(expanded_gate_output.shape)
        repeat_gate_output = tf.tile(expanded_gate_output, [1,model_config.mmoe_units,1])
        print(repeat_gate_output.shape)
        weighted_expert_output = expert_outputs * repeat_gate_output
        print(weighted_expert_output.shape)
        sum_expert_output = tf.reduce_sum(weighted_expert_output, axis=2)
        print(sum_expert_output.shape)
        final_outputs.append(sum_expert_output)
        
    # ------tower full connect layers------
    for index in range(model_config.tasks_num):
        with tf.variable_scope("tower-layers_"+str(index), reuse=tf.AUTO_REUSE):
            layers = model_config.tower_layers
            tower_input = final_outputs[index]
            #print(layers)
            # predict_scores = batch_norm_layer(predict_scores, mode == tf.estimator.ModeKeys.TRAIN, "bn")
            for i, layer_size in enumerate(layers + [1]):
                #print(layer_size)
                # predict_scores = batch_norm_layer(predict_scores, mode == tf.estimator.ModeKeys.TRAIN, "bn"+str(i))
                predict_scores = tf.layers.dense(inputs=tower_input,
                                                units=layer_size,
                                                activation=tf.nn.relu if i < len(layers) else None,
                                                use_bias=True,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                                trainable=True
                                                )
                #tf.logging.info("Shape of predict_scores:{}".format(str(predict_scores.shape)))
        predict_scores = tf.reshape(predict_scores, [-1])
        #end_points["debug_1"] = query_ids
        #end_points["debug_2"] = title_ids
        end_points[title_source+"_logits_"+str(index)] = predict_scores
        end_points[title_source+"_probabilities_"+str(index)] = tf.nn.sigmoid(predict_scores)
    print("create network_fn finish")
    return end_points

def loss_fn_binary_crossentropy(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    
def loss_fn_mean_squared_error(labels, logits):
    return tf.reduce_mean(tf.square(labels - logits))

def loss_fn_pair(pos_logits,neg_logits,pos_lab,neg_lab):
    label = tf.where(pos_lab > neg_lab, x=tf.ones_like(pos_lab, dtype = tf.float32), y=tf.zeros_like(neg_lab, dtype = tf.float32))
    prediction = pos_logits - neg_logits
    prediction = tf.minimum(tf.maximum(tf.sigmoid(prediction), 1e-6), 1 - 1e-6);
    return tf.reduce_mean(-label * tf.log(prediction) - (1 - label) * tf.log(1 - prediction))

def loss_fn_mmoe(labels, end_points):
    loss_relev = loss_fn_binary_crossentropy(labels['pointwise_lab'], end_points['pointwise_logits_0'])
    loss_ordcnt = loss_fn_pair(end_points['pairwise_pos_logits_1'], end_points['pairwise_neg_logits_1'], labels['pairwise_pos_lab'], labels['pairwise_neg_lab'])
    
    return loss_relev, loss_ordcnt

def calc_focal_loss(labels, logits, gamma=2):
    prob = tf.nn.sigmoid(logits)
    pos_weight = tf.identity(tf.pow(1-prob, gamma))
    neg_weight = tf.identity(tf.pow(prob, gamma))
    sum_weight = tf.reduce_sum(tf.multiply(pos_weight, labels) + tf.multiply(neg_weight, 1-labels))
    pos_weight = tf.clip_by_value(tf.divide(pos_weight, sum_weight+1e-10), 0.1, 4)
    neg_weight = tf.clip_by_value(tf.divide(neg_weight, sum_weight+1e-10), 0.1, 4)
    #pos = -tf.multiply( tf.stop_gradient(pos_weight), tf.multiply(labels, tf.log(tf.clip_by_value(prob, 0.1, 0.9)) ))
    #neg = -tf.multiply( tf.stop_gradient(neg_weight), tf.multiply(1-labels, tf.log(tf.clip_by_value(1-prob, 0.1, 0.9))) )
    pos = -tf.multiply( tf.stop_gradient(pos_weight), tf.multiply(labels, tf.log( prob ) ))
    neg = -tf.multiply( tf.stop_gradient(neg_weight), tf.multiply(1-labels, tf.log( 1-prob )) )
    return tf.reduce_sum(pos+neg)


def calc_inner_product(matrix_1, matrix_2):
    '''
    :param matrix_1: [None, length_1, size]
    :param matrix_2: [None, length_2, size]
    :return: [None, length_1, length_2]
    '''
    return tf.matmul(matrix_1, matrix_2, transpose_b=True)


def get_optimizer(learning_rate):
    # return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #return tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    #return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #return tf.train.FtrlOptimizer(learning_rate)
    #return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #return optimizer.MaskedAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def define_metrics(labels, eval_logits):
    
    # task_relev
    y_predict_relev = tf.concat([ele["pointwise_probabilities_0"] for ele in eval_logits], 0)
    y_predict_relev_2 = tf.where(y_predict_relev < 0.5, x=tf.zeros_like(y_predict_relev, dtype = tf.float32), y=tf.ones_like(y_predict_relev, dtype = tf.float32))
    accuracy_relev = tf.metrics.accuracy(tf.cast(tf.greater(labels['pointwise_lab'], 0.5), tf.float32), y_predict_relev_2)
    sum_auc_relev = tf.metrics.auc(tf.cast(tf.greater(labels['pointwise_lab'], 0.5), tf.float32), y_predict_relev)
    # task_ordcnt
    y_predict_pos = tf.concat([ele["pairwise_pos_logits_1"] for ele in eval_logits], 0)
    y_predict_neg = tf.concat([ele["pairwise_neg_logits_1"] for ele in eval_logits], 0)
    
    y_predict_diff = tf.where(y_predict_pos > y_predict_neg, x=tf.ones_like(y_predict_pos, dtype = tf.float32), y=tf.zeros_like(y_predict_neg, dtype = tf.float32))
    y_label = tf.where(labels['pairwise_pos_lab'] > labels['pairwise_neg_lab'] , x=tf.ones_like(y_predict_pos, dtype = tf.float32), y=tf.zeros_like(y_predict_neg, dtype = tf.float32))
    accuracy_pairwise = tf.metrics.accuracy(y_label, y_predict_diff)
    
    y_predict_pairwise = tf.nn.sigmoid(y_predict_pos - y_predict_neg)
    y_label_a, y_label_b = tf.split(y_label, num_or_size_splits=2, axis=0)
    y_predict_a, y_predict_b = tf.split(y_predict_pairwise, num_or_size_splits=2, axis=0)
    y_label_auc = tf.concat([y_label_a, 1 - y_label_b], 0)
    y_predict_auc = tf.concat([y_predict_a, 1.0 - y_predict_b], 0)
    sum_auc_pairwise = tf.metrics.auc(y_label_auc, y_predict_auc)
    
    metrics = \
        {
            "relev_accuracy": accuracy_relev,
            "relev_auc": sum_auc_relev,
            "pairwise_accuracy": accuracy_pairwise,
            "pairwise_auc": sum_auc_pairwise
        }
    tf.summary.scalar('relev_accuracy', accuracy_relev[1])
    tf.summary.scalar('relev_auc', sum_auc_relev[1])
    tf.summary.scalar('pairwise_accuracy', accuracy_pairwise[1])
    tf.summary.scalar('pairwise_auc', sum_auc_pairwise[1])    
    
    return metrics

def model_fn(features, labels, mode, params):
    """Build Model function f(x) for Estimator."""
    learning_rate = model_config.learning_rate
    gpus_list = model_config.gpus_list
    if len(gpus_list)==0 and len(params.get('gpus_list',"")) > 1 and mode == tf.estimator.ModeKeys.TRAIN:
        gpus_list = [int(ele) for ele in params['gpus_list'].split(",")]
    features = format_features(features)
    spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        end_points = {}
        end_points = network_fn(features, labels, mode, params, end_points, 'pointwise', 'pointwise')
        end_points = network_fn(features, labels, mode, params, end_points, 'pairwise', 'pairwise_pos')
        end_points = network_fn(features, labels, mode, params, end_points, 'pairwise', 'pairwise_neg')
        
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=end_points)
    else:
        # ------bulid optimizer------
        optimizer = get_optimizer(learning_rate)
        global_step = tf.train.get_or_create_global_step()

        # tower model
        eval_logits = []
        total_loss = 0
        if len(gpus_list) > 1:
            features_split = [{} for i in range(len(gpus_list))]
            labels_split = []
            for name in features.keys():
                value = features[name]
                batch_size = tf.shape(value)[0]
                split_size = batch_size // len(gpus_list)
                splits = [split_size, ] * (len(gpus_list) - 1)
                splits.append(batch_size - split_size * (len(gpus_list) - 1))
                # Split the features and labels
                value_split = tf.split(value, splits, axis=0)
                for i in range(len(gpus_list)):
                    features_ele = features_split[i]
                    features_ele[name] = value_split[i]
                    features_split[i] = features_ele
                if len(labels_split) <= 0:
                    labels_split = tf.split(labels, splits, axis=0)

            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(len(gpus_list)):
                    gpu_index = gpus_list[i]
                    with tf.device('/gpu:%d' % gpu_index):
                        with tf.name_scope('%s_%d' % ("tower-gpu", gpu_index)) as scope:
                            # model and loss
                            logits, end_points = network_fn(features_split[i], labels_split[i], mode, model_config)
                            losses = loss_fn(labels_split[i], logits)
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                            updates_op = tf.group(*update_ops)
                            with tf.control_dependencies([updates_op]):
                                # losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                                total_loss += losses / len(gpus_list)
                            # reuse var
                            tf.get_variable_scope().reuse_variables()
                            # grad compute
                            grads = optimizer.compute_gradients(losses)
                            tower_grads.append(grads)
                            # for eval metric ops
                            eval_logits.append(end_points)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = average_gradients(tower_grads)

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

            # Track the moving averages of all trainable variables.
            # variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
            # variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, grads)

        else:
            end_points = {}
            end_points = network_fn(features, labels, mode, params, end_points, 'pointwise', 'pointwise')
            end_points = network_fn(features, labels, mode, params, end_points, 'pairwise', 'pairwise_pos')
            end_points = network_fn(features, labels, mode, params, end_points, 'pairwise', 'pairwise_neg')
            
            if model_config.multi_type == 'pointwise':
                loss_relev = loss_fn_binary_crossentropy(labels['pointwise_lab'], end_points['pointwise_logits_0'])
                total_loss = loss_relev
            elif model_config.multi_type == 'pairwise':
                loss_ordcnt = loss_fn_pair(end_points['pairwise_pos_logits_1'], end_points['pairwise_neg_logits_1'], labels['pairwise_pos_lab'], labels['pairwise_neg_lab'])
                total_loss = loss_ordcnt                                
            else:
                loss_relev = loss_fn_binary_crossentropy(labels['pointwise_lab'], end_points['pointwise_logits_0'])
                loss_ordcnt = loss_fn_pair(end_points['pairwise_pos_logits_1'], end_points['pairwise_neg_logits_1'], labels['pairwise_pos_lab'], labels['pairwise_neg_lab'])
                total_loss = 0.8 * loss_relev + 0.2 * loss_ordcnt

            eval_logits.append(end_points)
            train_op = optimizer.minimize(total_loss, global_step=global_step)
            # relev_train_op = optimizer.minimize(loss_relev, global_step=global_step)
            # ordcnt_train_op = optimizer.minimize(loss_ordcnt, global_step=global_step)
        metrics = define_metrics(labels, eval_logits)
        if mode == tf.estimator.ModeKeys.TRAIN:
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=metrics)

    return spec


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
