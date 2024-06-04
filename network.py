# -*- coding:utf-8 -*-
import logging
import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

import warnings
import nets

warnings.filterwarnings("ignore")
# tf.compat.v1.enable_eager_execution()
# tf.enable_eager_execution()

epsilon = 1e-9


class Model(object):

    def __init__(self, settings):
        self.max_length = settings.max_length
        self.n_items = settings.n_items
        self.n_users = settings.n_users
        self.user_dim = settings.user_dim
        self.item_id_dim = settings.item_dim
        self.cate_dim = settings.cate_dim
        self.item_dim = self.item_id_dim + 2 * self.cate_dim
        self.lr_reg = settings.lr_reg
        self.gnn_layer_chose = settings.gnn_layer_chose
        # het_gcn's parameters
        self.metric_heads = settings.metric_heads  # 1
        self.relative_threshold = settings.relative_threshold  # 0.3
        self.pool_layers = settings.pool_layers  # 2

        self.cons_w = settings.cons_w
        self.tau = settings.tau

        self.b_to_c = settings.b_to_c
        self.b_to_b = settings.b_to_b
        self.c_to_c = settings.c_to_c
        self.c_to_b = settings.c_to_b

        self.pos_w = settings.pos_w
        self.neg_w = settings.neg_w
        self.like_w = settings.like_w

        self.model_name = settings.model_name
        self.predict_source = settings.predict_source
        self.loss_flag = settings.loss_flag

        self.dnn_size = settings.dnn_size

        self.interest_dim = settings.interest_dim
        self.hidden_size = self.interest_dim

        self.batch_size = settings.batch_size
        self.dataset_flag = settings.dataset_flag

        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(settings.learning_rate,
                                                              self.global_step,
                                                              settings.decay_steps,
                                                              settings.decay_rate,
                                                              staircase=True))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.item_embedding = tf.get_variable('item_embedding', [self.n_items, self.item_id_dim], initializer=tf.glorot_normal_initializer(), trainable=True)
        # self.user_embedding = tf.get_variable('user_embedding', [self.n_users, self.user_dim], initializer=tf.glorot_normal_initializer(), trainable=True)
        self.category_embedding = tf.get_variable('category_embedding', [31, self.cate_dim], initializer=tf.glorot_normal_initializer(), trainable=True)
        # self.category_embedding = tf.concat([tf.zeros((1, self.cate_dim)),category_embedding], axis=0)
        tag_embedding = tf.get_variable('tag_embedding', [59, self.cate_dim], initializer=tf.glorot_normal_initializer(), trainable=True)
        self.tag_embedding = tf.concat([tf.zeros((1, self.cate_dim)), tag_embedding], axis=0)     # tag=0表示没有tag，因此用0来表示

        # placeholders
        self.tst = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('Inputs'):
            self.seq_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_inputs')  # all history  seq_input
            self.seq_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='seq_mask_input')

            self.seq_cate_input = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_cate_input')
            self.seq_tag_input = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_tag_input')

            self.seq_clickmask_input = tf.placeholder(tf.bool, [self.batch_size, self.max_length], name='seq_clickmask_input')  # click mask
            self.seq_likemask_input = tf.placeholder(tf.bool, [self.batch_size, self.max_length], name='seq_likemask_input')  # like mask

            self.seq_pos_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_pos_inputs')
            self.seq_pos_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='seq_pos_mask_inputs')
            self.seq_pos_idxs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_pos_idxs')

            self.seq_neg_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_pos_inputs')
            self.seq_neg_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='seq_pos_mask_inputs')
            self.seq_neg_idxs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_neg_idxs')

            self.y_inputs = tf.placeholder(tf.float32, [self.batch_size], name='y_input')  # label
            self.item_inputs = tf.placeholder(tf.int32, [self.batch_size], name='item_inputs')  # item_input
            self.user_id_inputs = tf.placeholder(tf.int32, [self.batch_size], name='user_id_inputs')  # user_ids
            self.item_cate_input = tf.placeholder(tf.int32, [self.batch_size], name='item_cate_input')
            self.item_tag_input = tf.placeholder(tf.int32, [self.batch_size], name='item_tag_input')

            self.like_label = tf.placeholder(tf.float32, [self.batch_size], name='like_label')  # like_label
            self.watch_ratio_label = tf.placeholder(tf.float32, [self.batch_size], name='watch_ratio_label')  # like_label
            self.complete_label = tf.placeholder(tf.float32, [self.batch_size], name='complete_label')  # complete_label

            self.like_gate = tf.placeholder(tf.bool, [self.batch_size], name='like_gate')

            item = tf.nn.embedding_lookup(self.item_embedding, self.item_inputs)
            item_cate_feat = tf.nn.embedding_lookup(self.category_embedding, self.item_cate_input)
            item_tag_feat = tf.nn.embedding_lookup(self.tag_embedding, self.item_tag_input)
            self.item = tf.concat([item, item_cate_feat, item_tag_feat], axis=-1)

            seq_features = tf.nn.embedding_lookup(self.item_embedding, self.seq_inputs)
            seq_cate_feat = tf.nn.embedding_lookup(self.category_embedding, self.seq_cate_input)
            seq_tag_feat = tf.nn.embedding_lookup(self.tag_embedding, self.seq_tag_input)
            self.seq_features = tf.concat([seq_features, seq_cate_feat, seq_tag_feat], axis=-1)

            # self.user_feature = tf.nn.embedding_lookup(self.user_embedding, self.user_id_inputs)

            self.seq_pos_features = tf.nn.embedding_lookup(self.item_embedding, self.seq_pos_inputs)
            self.seq_neg_features = tf.nn.embedding_lookup(self.item_embedding, self.seq_neg_inputs)

            key_masks = tf.sequence_mask(self.seq_mask_inputs, self.max_length)  # [B, T]
            self.unclick_mask = tf.logical_and(tf.logical_not(self.seq_clickmask_input), key_masks)

        if self.model_name in ["het_lstm", "het_graph", "het_lstm_graph"]:
            if self.model_name == "het_lstm" or self.model_name == "het_lstm_graph":
                with tf.name_scope('het_lstm'), tf.variable_scope("het_lstm", reuse=tf.AUTO_REUSE):
                    logging.info("het_lstm...")
                    rnn_outputs, unclick_rnn_outputs, like_rnn_outputs = self.het_lstm(self.seq_features, self.seq_mask_inputs, tf.cast(self.seq_clickmask_input, tf.float32))
                    self.pos_user_emb = rnn_outputs

            if self.model_name == "het_graph" or self.model_name == "het_lstm_graph":
                with tf.name_scope('interest_graph'):

                    key_masks = tf.sequence_mask(self.seq_mask_inputs, tf.shape(self.seq_features)[1])  # [B, T]
                    A_browse = self.calculate_simi_matrix(self.seq_features, key_masks)  # browse 的邻接矩阵
                    A_click = self.calculate_simi_matrix(self.seq_features, self.seq_clickmask_input)  # [B,L,L]    click的邻接矩阵
                    A_browse = self.adj_matrix_normalize(A_browse)
                    A_click = self.adj_matrix_normalize(A_click)
                    A_cross = tf.eye(self.max_length) * tf.cast(tf.expand_dims(self.seq_clickmask_input, axis=-1), tf.float32)  # [B,L,L]*[B,L,1]

                with tf.name_scope('interest_fusion'),tf.variable_scope("interest_fusion", reuse=tf.AUTO_REUSE):
                    key_masks = tf.sequence_mask(self.seq_mask_inputs, self.max_length)
                    X_browse = self.seq_features * tf.cast(tf.expand_dims(key_masks, -1), tf.float32)  # [B,L,D]*[B,L,1]
                    X_click = self.seq_features * tf.cast(tf.expand_dims(self.seq_clickmask_input, -1), tf.float32)  # [B,L,d]*[B,L,1]
                    X_browse_all = [X_browse]
                    X_click_all = [X_click]
                    for l in range(self.pool_layers):
                        # reuse = False if l == 0 else True
                        X_browse, X_click = self._interest_fusion(X_browse, X_click, A_browse, A_click, A_cross)  # [B,L,D]
                        X_browse_all += [X_browse]
                        X_click_all += [X_click]
                    if self.gnn_layer_chose == "all":
                        self.graph_browse_vector = tf.reduce_mean(tf.stack(X_browse_all, 0), 0)
                        self.graph_click_vector = tf.reduce_mean(tf.stack(X_click_all, 0), 0)
                    elif self.gnn_layer_chose == "last":
                        self.graph_browse_vector = X_browse
                        self.graph_click_vector = X_click

                    # 加一层lstm
                    # click_seq = tf.batch_gather(X_click, self.seq_neg_idxs)
                    # unclick_seq = tf.batch_gather(X_browse, self.seq_neg_idxs)
                    # with tf.name_scope('pos'), tf.variable_scope('pos'):
                    #     graph_outputs = self.vanilla_lstm(click_seq, self.seq_pos_mask_inputs)
                    # with tf.name_scope('neg'), tf.variable_scope('neg'):
                    #     unclick_graph_outputs = self.vanilla_lstm(unclick_seq, self.seq_neg_mask_inputs)


                    graph_outputs = self.vanilla_attention_boolmask(self.item, self.graph_click_vector, self.seq_clickmask_input)
                    unclick_graph_outputs = self.vanilla_attention_boolmask(self.item, self.graph_browse_vector, self.unclick_mask)

            if self.model_name == "het_lstm":
                pos_outputs, neg_outputs = rnn_outputs, unclick_rnn_outputs
            elif self.model_name == "het_graph":
                pos_outputs, neg_outputs = graph_outputs, unclick_graph_outputs
            elif self.model_name == "het_lstm_graph":
                with tf.name_scope('crossviewfuse'),tf.variable_scope("crossviewfuse", reuse=tf.AUTO_REUSE):
                    gate_b = tf.sigmoid(tf.layers.dense(self.rnn_items_vector, units=self.item_dim, use_bias=False, name='fuse1') + \
                                        tf.layers.dense(self.graph_browse_vector, units=self.item_dim, use_bias=False, name='fuse2'))
                    all_browse_seq = tf.multiply(gate_b, self.rnn_items_vector) + tf.multiply(1 - gate_b, self.graph_browse_vector)

                    gate_c = tf.sigmoid(tf.layers.dense(self.rnn_items_vector, units=self.item_dim, use_bias=False, name='fuse3') + \
                                        tf.layers.dense(self.graph_click_vector, units=self.item_dim, use_bias=False, name='fuse4'))
                    all_click_seq = tf.multiply(gate_c, self.rnn_items_vector) + tf.multiply(1 - gate_c, self.graph_click_vector)

                    pos_outputs = self.vanilla_attention_boolmask(self.item, all_click_seq, self.seq_clickmask_input)
                    neg_outputs = self.vanilla_attention_boolmask(self.item, all_browse_seq, self.unclick_mask)


            with tf.name_scope('predict'):
                if self.predict_source == "pos":  # pos   pos_neg   neg   like   like_pos   like_pos_neg
                    self.user_emb = pos_outputs
                elif self.predict_source == "neg":
                    self.user_emb = neg_outputs
                elif self.predict_source == "like":
                    self.user_emb = like_rnn_outputs
                elif self.predict_source == "pos_neg":
                    self.user_emb = self.pos_w * pos_outputs + self.neg_w * neg_outputs
                elif self.predict_source == "like_pos":
                    self.user_emb = self.like_w * like_rnn_outputs + self.pos_w * pos_outputs
                elif self.predict_source == "like_pos_neg":
                    self.user_emb = self.pos_w * pos_outputs + self.neg_w * neg_outputs + self.like_w * like_rnn_outputs

                self.joint_output = self.predict_score_new(self.user_emb)

        self.joint_evaoutput = tf.nn.sigmoid(self.joint_output)

        l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'embedding' not in v.name])
        self.joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.joint_output, labels=self.y_inputs)) + l2_norm * self.lr_reg
        #                   0.0000005

        if self.model_name == "het_lstm_graph" and self.loss_flag == "contrastive":
            with tf.name_scope('contrastive_loss'):
                self.rnn_items_vector = self.contras_proj(self.rnn_items_vector, reuse=False)
                self.graph_browse_vector = self.contras_proj(self.graph_browse_vector, reuse=True)
                self.graph_click_vector = self.contras_proj(self.graph_click_vector, reuse=True)

                # key_masks = tf.sequence_mask(self.seq_mask_inputs, self.max_length)  # [B, T]
                # unclick_mask = tf.logical_and(tf.logical_not(self.seq_clickmask_input), key_masks)

                click_contras_loss = self.contras_loss(self.rnn_items_vector, self.graph_click_vector, self.seq_clickmask_input)
                unclick_contras_loss = self.contras_loss(self.rnn_items_vector, self.graph_browse_vector, self.unclick_mask)

                self.joint_loss = self.joint_loss + self.cons_w * (click_contras_loss + unclick_contras_loss) / 2

        
        for v in tf.trainable_variables():
            logging.info(v.name)

        grads_and_vars = self.optimizer.compute_gradients(self.joint_loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

    def contras_loss(self, rnn_embs, graph_embs, embs_mask):

        rnn_embs = tf.nn.l2_normalize(rnn_embs, axis=-1)  # tf.nn.l2_normalize(, axis=1)
        graph_embs = tf.nn.l2_normalize(graph_embs, axis=-1)  # tf.nn.l2_normalize(, axis=1)

        pos_scores = tf.reduce_sum(rnn_embs * graph_embs, axis=-1)  # [B,L]
        paddings = tf.ones_like(pos_scores) * (-2 ** 32 + 1)
        pos_scores = tf.where(embs_mask, pos_scores, paddings)  # [B,L]

        neg_scores = rnn_embs @ tf.transpose(graph_embs, perm=[0, 2, 1])  # [B,L,L]
        paddings = tf.ones_like(neg_scores) * (-2 ** 32 + 1)
        masks_matrix = tf.cast(embs_mask, tf.float32)
        masks_matrix = tf.expand_dims(masks_matrix, -1) * tf.expand_dims(masks_matrix, -2)  # [B,L,L]*[B,L,1]*[B,1,L] = [B,L,L]
        masks_matrix = tf.cast(masks_matrix, tf.bool)
        neg_scores = tf.where(masks_matrix, neg_scores, paddings)

        posScore = tf.exp(pos_scores / self.tau)  # [B,L,d] * [B,L,d] = [B,L]
        negScore = tf.reduce_sum(tf.exp(neg_scores / self.tau), axis=-1)  # [B,L,D] [B,D,L]=[B,L,L] -> [B,L]
        rate = tf.reduce_sum(posScore, axis=-1) / (tf.reduce_sum(negScore, axis=-1) + epsilon)  ##[B]
        # rate = tf.Print(rate,["rate",rate],summarize =10)
        contras_loss = tf.reduce_mean(-tf.log(rate + epsilon))
        # contras_loss = tf.Print(contras_loss,["contras_loss",contras_loss],summarize =10)

        return contras_loss

    def contras_proj(self, X, reuse):
        output = tf.layers.dense(X, activation=tf.nn.relu, units=self.item_dim, use_bias=True, name='contras_proj1', reuse=reuse)
        output = tf.layers.dense(output, activation=None, units=self.item_dim, use_bias=True, name='contras_proj2', reuse=reuse)
        return output

    def calculate_simi_matrix(self, X, key_masks):
        float_mask = tf.cast(key_masks, tf.float32)
        S = []
        for i in range(self.metric_heads):
            # weighted cosine similarity
            self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)  # [1,D]
            X_fts = X * tf.expand_dims(self.weighted_tensor, 0)  # [B,L,D] * [1,1,D] = [B,L,D]
            X_fts = tf.nn.l2_normalize(X_fts, dim=2)
            S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0, 2, 1)))  # B*L*L     [B,L,D]*[B,D,L] =[B,L,L]
            S_one = S_one * tf.expand_dims(float_mask, -1) * tf.expand_dims(float_mask, -2)  # [B,L,L]*[B,L,1]*[B,1,L] = [B,L,L]
            # min-max normalization for mask
            # S_min = tf.reduce_min(S_one, -1, keepdims=True)
            # S_max = tf.reduce_max(S_one, -1, keepdims=True)
            # S_one = (S_one - S_min) / (S_max - S_min + epsilon)  # 【规划到0，1中】
            S += [S_one]
        S = tf.reduce_mean(tf.stack(S, 0), 0)
        # mask invalid nodes
        S = S * tf.expand_dims(float_mask, -1) * tf.expand_dims(float_mask, -2)  # [B,L,L]*[B,L,1]*[B,1,L] = [B,L,L]
        S = tf.where(S < 0, tf.zeros_like(S), S)
        ## Graph sparsification via seted sparseness
        S_flatten = tf.reshape(S, [tf.shape(S)[0], -1])  # [B,L*L]
        sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L              【B,L*L】
        # relative ranking strategy of the entire graph
        num_edges = tf.cast(tf.count_nonzero(S, [1, 2]), tf.float32)  # B
        to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
        threshold_score = tf.batch_gather(sorted_S_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1))  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
        A = tf.cast(tf.greater(S, tf.expand_dims(threshold_score, -1)), tf.float32)
        return A

    def adj_matrix_normalize(self, A):
        A_bool = tf.cast(tf.greater(A, 0), A.dtype)  # [B,L,L]
        A_bool = A_bool * (tf.ones([A.shape.as_list()[1], A.shape.as_list()[-1]]) - tf.eye(A.shape.as_list()[-1])) + tf.eye(A.shape.as_list()[-1])  # [B,L,L]
        D = tf.reduce_sum(A_bool, axis=-1)  # B*L
        D = tf.sqrt(D)[:, None] + K.epsilon()  # B*1*L
        A = (A_bool / D) / tf.transpose(D, perm=(0, 2, 1))  # B*L*L / B*1*L / B*L*1
        return A

    def _gcn_proj(self,X):
        output = tf.layers.dense(X, activation=tf.nn.tanh, units=self.item_dim, use_bias=False, name='gcn_proj1')
        output = tf.layers.dense(output, activation=None, units=1, use_bias=False, name='gcn_proj2')
        return output

    def _interest_fusion(self, X_browse, X_click, A_browse, A_click, A_cross):
        # browse
        X_bb = tf.matmul(A_browse, X_browse)  # B*L*L x B*L*D -> B*L*D
        X_bb = tf.layers.dense(X_bb, activation=tf.nn.leaky_relu,units=X_browse.shape.as_list()[-1], use_bias=False, name='fuse_proj1')
        X_bc = tf.matmul(A_cross, X_click)
        X_bc = tf.layers.dense(X_bc, activation=tf.nn.leaky_relu,units=X_bc.shape.as_list()[-1], use_bias=False, name='fuse_proj2')

        alpha1,alpha2 = self._gcn_proj(tf.concat([X_bb,X_browse],-1)),self._gcn_proj(tf.concat([X_bc,X_browse],-1))
        alpha1 = tf.math.exp(alpha1)/(tf.math.exp(alpha1)+tf.math.exp(alpha2)+epsilon)
        alpha2 = 1-alpha1
        X_b = alpha1 * X_bb + alpha2 * X_bc
        X_b = tf.nn.dropout(X_b, self.keep_prob)  # [BS,max_len,64]

        # click
        X_cc = tf.matmul(A_click, X_click)  # B*L*L x B*L*F -> B*L*F
        X_cc = tf.layers.dense(X_cc, activation=tf.nn.leaky_relu,units=X_click.shape.as_list()[-1], use_bias=False, name='fuse_proj3')
        X_cb = tf.matmul(A_cross, X_browse)
        X_cb = tf.layers.dense(X_cb, activation=tf.nn.leaky_relu,units=X_cb.shape.as_list()[-1], use_bias=False, name='fuse_proj4')

        alpha1, alpha2 = self._gcn_proj(tf.concat([X_cc,X_click],-1)), self._gcn_proj(tf.concat([X_cb,X_click],-1))
        alpha1 = tf.math.exp(alpha1) / (tf.math.exp(alpha1) + tf.math.exp(alpha2) + epsilon)
        alpha2 = 1 - alpha1
        X_c = alpha1 * X_cc + alpha2 * X_cb
        X_c = tf.nn.dropout(X_c, self.keep_prob)  # [BS,max_len,64]

        return X_b, X_c
    def cell_hetlstm_forward(self, xt, H_c, H_v, s_c, s_v, g):
        n_hidden = self.hidden_size
        f_c = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xfc') + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hfc'))
        i_c = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xic') + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hic'))
        o_c = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xoc') + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hoc'))
        g_c = tf.tanh(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xgc') + tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hgc'))
        # s_c = tf.multiply(1 - g, tf.multiply(f_c, s_c)) + tf.multiply(i_c, g_c)
        s_c = tf.multiply(f_c, s_c) + tf.multiply(i_c, g_c)
        H_c = tf.multiply(o_c, tf.tanh(s_c))

        f_v = tf.sigmoid(tf.layers.dense(H_c, units=n_hidden, use_bias=True, name='xfv') + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hfv'))
        i_v = tf.sigmoid(tf.layers.dense(H_c, units=n_hidden, use_bias=True, name='xiv') + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hiv'))
        o_v = tf.sigmoid(tf.layers.dense(H_c, units=n_hidden, use_bias=True, name='xov') + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hov'))
        g_v = tf.tanh(tf.layers.dense(H_c, units=n_hidden, use_bias=True, name='xgv') + tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hgv'))
        s_v = tf.multiply(1 - g, s_v) + tf.multiply(g, tf.multiply(f_v, s_v) + tf.multiply(i_v, g_v))
        H_v = tf.multiply(1 - g, H_v) + tf.multiply(g, tf.multiply(o_v, tf.tanh(s_v)))
        H_c = tf.multiply(1 - g, H_c) + tf.multiply(g, H_v)  # 如果是click,H_c保存上一层的hidden state,

        return H_c, H_v, s_c, s_v

    def het_lstm(self, inputs, inputs_mask, click_mask):

        inputs = tf.unstack(inputs, axis=1)  # [max_len,bs,64]
        g = tf.unstack(click_mask, axis=1)  # [max_len,bs]
        g = tf.expand_dims(g, axis=-1)

        n_hidden = self.hidden_size

        # keys = nets.dense(item_block_emb, self.interest_dim, ["w_l2"], keep_prob=self.keep_prob, activation=None)
        H_c = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
        H_v = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
        s_c = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
        s_v = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)

        H_c, H_v, s_c, s_v = self.cell_hetlstm_forward(inputs[0], H_c, H_v, s_c, s_v, g[0])
        rnn_outputs = H_c
        rnn_outputs = tf.reshape(rnn_outputs, [-1, 1, self.hidden_size])

        for i in range(1, self.max_length):
            H_c, H_v, s_c, s_v = self.cell_hetlstm_forward(inputs[i], H_c, H_v, s_c, s_v, g[i])
            rnn_outputs = tf.concat([rnn_outputs, tf.reshape(H_c, [-1, 1, self.hidden_size])], 1)  # 只要保存H_c就好了
        rnn_outputs = tf.nn.dropout(rnn_outputs, self.keep_prob)  # [BS,max_len,64]
        self.rnn_items_vector = rnn_outputs  # [B,L,D]

        click_rnn_outputs = self.vanilla_attention_boolmask(self.item, rnn_outputs, self.seq_clickmask_input)
        # key_masks = tf.sequence_mask(self.seq_mask_inputs, tf.shape(inputs)[0])  # [B, T]
        # unclick_mask = tf.logical_and(tf.logical_not(self.seq_clickmask_input), key_masks)
        unclick_rnn_outputs = self.vanilla_attention_boolmask(self.item, rnn_outputs, self.unclick_mask)
        like_rnn_outputs = self.vanilla_attention_boolmask(self.item, rnn_outputs, self.seq_likemask_input)

        return click_rnn_outputs, unclick_rnn_outputs, like_rnn_outputs

    def vanilla_lstm(self, inputs, inputs_mask):
        # lstm + attention
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        rnn_outputs, states1 = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                 inputs=inputs,
                                                 sequence_length=inputs_mask,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32)
        last_h = states1.h  # [B,hidden]

        # rnn_outputs = tf.nn.dropout(rnn_outputs, self.keep_prob)
        # rnn_outputs = self.vanilla_attention_mask(self.item, rnn_outputs, inputs_mask)
        rnn_outputs = last_h  # [B,hidden]

        return rnn_outputs

    def predict_score_new(self, interest_emb):
        output = tf.layers.dense(tf.concat([interest_emb, self.item], axis=1), activation=tf.nn.relu, units=self.dnn_size, use_bias=True, name='dnn1')
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.layers.dense(output, activation=None, units=1, use_bias=True, name='dnn2')
        return tf.reshape(output, [-1])

    def vanilla_attention(self, queries, keys):  # query [B,64],  keys [B,5,64]
        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.interest_dim])

    def vanilla_attention_mask(self, queries, keys, keys_length):  # keys [B,T,H]
        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.item_dim])

    def vanilla_attention_boolmask(self, queries, keys, key_masks):  # keys [B,T,H]   mask [B,T]
        queries = tf.layers.dense(queries, units=self.item_dim,use_bias=False, name='sequencerepresentation' )

        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Mask
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.item_dim])

    def vanilla_attention_qkv_mask(self, queries, keys, vals, keys_length):  # keys [B,T,H]
        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, vals)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.item_dim])

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def para_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.0001, shape=shape)
        return tf.Variable(initial, name=name)
