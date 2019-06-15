# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import math

class NAIS(object):
    def __init__(self, config, gd):
        self.config = config
        self.gd = gd
        self.dl = self.gd.dl
    
    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input  = tf.placeholder(tf.int32, shape = [None, None], name='user_input')
            self.num_idx = tf.placeholder(tf.float32, shape=[None,1], name='num_idx')
            self.item_input = tf.placeholder(tf.int32, shape=[None,1], name='item_input')
            self.labels = tf.placeholder(tf.float32, shape=[None,1], name='labels')
            self.time_input = tf.placeholder(tf.int32, shape=[None, None], name='time_input')
            self.otime_input = tf.placeholder(tf.int32, shape=[None, 1], name='otime_input')
            
    def _create_variables(self):
        with tf.name_scope('embeddings'):
            c1 = tf.Variable(tf.truncated_normal(shape=[self.dl.num_items, self.config.embedding_size], mean=0.0, stddev=0.01),\
                            name='c1', dtype = tf.float32)
            c2 = tf.constant(0.0, tf.float32, [1, self.config.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([c1, c2], 0 , name='emebedding_Q_') 
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.dl.num_items,self.config.embedding_size]), name='embediing_Q', \
                                     dtype=tf.float32)
#            t1 = tf.Variable(tf.truncated_normal(shape=[self.dl.numT+1, self.config.embedding_size], mean=0.0, stddev=0.01),\
#                            name='t1', dtype = tf.float32)
#            t2 = tf.constant(0.0, tf.float32, [1, self.config.embedding_size], name='t2')
#            self.embedding_T = tf.concat([t1, t2], 0 , name='emebedding_T')
            self.bias = tf.Variable(tf.zeros(self.dl.num_items), name='bias')
            
            #attention network variables
            self.W = tf.Variable(tf.truncated_normal(shape=[self.config.embedding_size, self.config.weight_size], mean=0.0, \
                            stddev=tf.sqrt(tf.div(2.0, self.config.weight_size + self.config.embedding_size))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.bias_b = tf.Variable(tf.truncated_normal(shape=[1, self.config.weight_size], mean=0.0, \
                stddev=tf.sqrt(tf.divide(2.0, self.config.weight_size + self.config.embedding_size))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            self.h = tf.Variable(tf.ones([self.config.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)
            
    def _attention_MLP(self, q_, num_idx, B):#q_:(b,n,2e)
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = self.config.embedding_size
            MLP_output = tf.matmul(tf.reshape(q_,[-1,r]), self.W) + self.bias_b #(b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            MLP_output = tf.nn.dropout(MLP_output, 0.5)#(b*n, w)
    #         fc_mean, fc_var = tf.nn.moments(MLP_output, axes=[0,1])
    #         scale = tf.Variable(tf.ones([r]))
    #         shift = tf.Variable(tf.zeros([r]))
    #         epsilon = 0.001
    #         MLP_output = tf.nn.batch_normalization(MLP_output, fc_mean, fc_var, \
    #                                                shift, scale, epsilon)
            
            MLP_output = tf.nn.relu( MLP_output )
            #添加一个dropout
            A_ = tf.reshape(tf.matmul(MLP_output, self.h),[b,n]) #(b*n, w) * (w, 1) => (None, 1) => (b, n)
            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            mask_mat = tf.sequence_mask(tf.reduce_sum(num_idx,1), maxlen = n, dtype = tf.float32) # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_,1, keepdims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.config.beta, tf.float32, [1]))
    
            A = tf.expand_dims(tf.div(exp_A_, exp_sum),2) # (b, n, 1)
    #         time_input = tf.expand_dims(time_input, 1)
    #         time_input = tf.tile(time_input, [1, n, 1])
    #         A = A + time_input
    
            #加入时间衰减
    #         B = tf.expand_dims(B, 2)
    #         A = A*B
          
            return tf.reduce_sum(A * self.embedding_q_, 1)    
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input) # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input) # (b, 1, e)
            self.embedding_p = self._attention_MLP(self.embedding_q_ * self.embedding_q, self.num_idx, self.time_input) #(b,2e)
            self.embedding_q = tf.reduce_sum(self.embedding_q, 1) #(b,2e)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, tf.constant(self.config.alpha, tf.float32, [1]))
        #     output = tf.expand_dims(tf.reduce_sum(embedding_p*embedding_q, 1),1) + bias_i
        #     output = tf.layers.dense(embedding_p*embedding_q, units=1, activation = None)
        #     output = tf.map_fn(lambda x:x*5.0 , output)
        #     output = tf.layers.dense(embedding_p*embedding_q, units=16, activation = tf.nn.sigmoid)
        #     output = tf.layers.dense(output, 1, None)

            self.output = tf.sigmoid(self.coeff*tf.expand_dims(tf.reduce_sum(self.embedding_p*self.embedding_q, 1),1) + self.bias_i)
    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.config.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.config.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                        self.config.eta_bilinear * tf.reduce_sum(tf.square(self.W))
    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        print("already build the computing graph...")
    def train_and_evaluate(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_count in range(self.config.epoches):
                #train
                train_begin = time.time()
                for train_data in self.gd.generateTrainData():
                    
                    user_input_data = np.array(train_data[0]).astype(np.int32)
                    num_idx_data= np.array(train_data[1]).astype(np.float32)
                    item_input_data= np.array(train_data[2]).astype(np.int32)
                    time_input_data = np.array(train_data[3]).astype(np.float32)
                    otime_input_data = np.array(train_data[4])
                    labels_data = np.array(train_data[5]).astype(np.float32)
                    
                    feed_dict = {self.user_input: user_input_data, self.num_idx: num_idx_data[:, np.newaxis], self.item_input: item_input_data[:, np.newaxis],self.time_input:time_input_data,
                                self.otime_input:otime_input_data[:,np.newaxis], self.labels: labels_data[:, np.newaxis]}
                    training_loss, _ = sess.run([self.loss, self.optimizer], feed_dict)
                train_time = time.time() - train_begin
                if epoch_count % self.config.verbose_count == 0:
                    #train loss
                    loss_begin = time.time()
                    train_loss = 0.0
                    batch_i = 0
                    for train_data in self.gd.generateTrainData():
                        batch_i +=1
                        user_input_data = np.array(train_data[0]).astype(np.int32)
                        num_idx_data= np.array(train_data[1]).astype(np.float32)
                        item_input_data= np.array(train_data[2]).astype(np.int32)
                        time_input_data = np.array(train_data[3]).astype(np.float32)
                        otime_input_data = np.array(train_data[4])
                        labels_data = np.array(train_data[5]).astype(np.float32)
                        
                        feed_dict = {self.user_input: user_input_data, self.num_idx: num_idx_data[:, np.newaxis], self.item_input: item_input_data[:, np.newaxis],self.time_input:time_input_data,
                                self.otime_input:otime_input_data[:,np.newaxis], self.labels: labels_data[:, np.newaxis]}
                        train_loss += sess.run(self.loss, feed_dict)
                    train_loss = train_loss/batch_i
                    loss_time = time.time() - loss_begin
                                         
                    eval_begin = time.time() 
                    hits, ndcgs, losses = [],[],[]
                    for test_data in self.gd.generateTestData():
                        user_input_data = np.array(test_data[0]).astype(np.int32)
                        num_idx_data= np.array(test_data[1]).astype(np.float32)
                        item_input_data= np.array(test_data[2]).astype(np.int32)
                        time_input_data = np.array(test_data[3]).astype(np.float32)
                        otime_input_data = np.array(test_data[4])
                        labels_data = np.array(test_data[5]).astype(np.float32)
                        
                        feed_dict = {self.user_input: user_input_data, self.num_idx: num_idx_data[:, np.newaxis], self.item_input: item_input_data[:, np.newaxis],self.time_input:time_input_data,
                                self.otime_input:otime_input_data[:,np.newaxis], self.labels: labels_data[:, np.newaxis]}
                        predictions,test_loss = sess.run([self.output,self.loss], feed_dict = feed_dict)
                        predictions = predictions.flatten()
        #                 min_ = np.min(predictions)
        #                 max_ = np.max(predictions)
        #                 print(min_, max_)
        #                 predictions = np.apply_along_axis(lambda x: 5*(x-min_)/(max_-min_), 0,predictions)
        
                        neg_predict, pos_predict = predictions[:-1], predictions[-1]
                        position = (neg_predict >= pos_predict).sum()
                        #print(position)
                        hr = position < 10
                        ndcg = math.log(2) / math.log(position+2) if hr else 0
                        hits.append(hr)
                        ndcgs.append(ndcg)  
                        losses.append(test_loss)
                    hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                    eval_time = time.time() - eval_begin
        #             print("Epoch %d [%.1fs ]:  train_loss = %.4f" % (
        #                     epoch_count,train_time, train_loss))    
                    print("Epoch %d [ %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                                epoch_count, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))

