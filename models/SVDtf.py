# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import math

class SVD(object):
    
    def __init__(self, config, dl):
        self.config = config
        self.dl = dl
        self.num_items = self.dl.num_items
        self.num_users = self.dl.num_users
        self.factors = self.config.factors
        # self.numT = self.dl.numT
        self.regU = self.config.regU
        self.B1 = self.config.regB1
        self.B2 = self.config.regB2
        self.regI = self.config.regI

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_idx = tf.placeholder(tf.int32, shape=[None,], name='user_idx')
            self.item_idx = tf.placeholder(tf.int32, shape=[None,], name = 'item_idx')
            self.labels = tf.placeholder(tf.float32, shape=[None,], name='labels')
    
    def _create_variables(self):
        with tf.name_scope('embeddings'):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.factors],mean=0.0,stddev=0.01), \
                                            dtype=tf.float32, name='embedding_P')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.factors],mean=0.0,stddev=0.01), \
                                            dtype=tf.float32, name='embedding_Q')
            
            self.U_bias = tf.Variable(tf.truncated_normal(shape=[self.num_users], stddev=0.005,mean=0.02), name='U_bias')
            
            self.V_bias = tf.Variable(tf.truncated_normal(shape=[self.num_items], stddev=0.005,mean=0.02), name='V_bias')
    def _create_inference(self):
        with tf.name_scope('inference'):
            self.embedding_pu = tf.nn.embedding_lookup(self.embedding_P, self.user_idx) #(b,k)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_idx) 
            self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.user_idx)
            self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.item_idx)
            self.output = tf.sigmoid(tf.reduce_sum(self.embedding_pu * self.embedding_q,1))
            self.output += self.U_bias_embed+self.V_bias_embed
            
    def _create_loss(self):
        with tf.name_scope("loss"):
            print(self.output.shape)
            self.loss = tf.losses.log_loss(self.labels, self.output) +  \
                                          self.regU*tf.reduce_sum(tf.square(self.embedding_P)) + \
                                        self.regI*tf.reduce_sum(tf.square(self.embedding_Q)) + \
                                        self.B1*tf.reduce_sum(tf.square(self.U_bias)) + \
                                        self.B2*tf.reduce_sum(tf.square(self.V_bias))
    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
        
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate, initial_accumulator_value=1e-8)
            self.train_U = self.optimizer.minimize(self.loss, var_list=[self.embedding_P, self.U_bias])
            self.train_V = self.optimizer.minimize(self.loss, var_list=[self.embedding_Q, self.V_bias])
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
                for train_data in self.dl.batch_gen():
                    
                    user_input_data = np.array(train_data[0]).astype(np.int32)
#                    time_input_data = np.array(train_data[1]).astype(np.float32)
                    item_input_data= np.array(train_data[1]).astype(np.int32)
                    
                    labels_data = np.array(train_data[2]).astype(np.float32)
                    
                    feed_dict = {self.user_idx: user_input_data,  self.item_idx: item_input_data, self.labels: labels_data}
                    sess.run(self.train_U, feed_dict)
                    sess.run(self.train_V, feed_dict)
                    
                    
                train_time = time.time() - train_begin
                if epoch_count % self.config.verbose_count ==0:
                    #train loss
                    loss_begin = time.time()
                    train_loss = 0.0
                    batch_i = 0
                    for train_data in self.self.dl.batch_gen():
                    
                        user_input_data = np.array(train_data[0]).astype(np.int32)
#                        time_input_data = np.array(train_data[1]).astype(np.float32)
                        item_input_data= np.array(train_data[1]).astype(np.int32)
                        
                        labels_data = np.array(train_data[2]).astype(np.float32)
                        
                        feed_dict = {self.user_idx: user_input_data,  self.item_idx: item_input_data, self.labels: labels_data}
                        train_loss += sess.run(self.loss, feed_dict)
                        batch_i+=1
                    train_loss = train_loss/batch_i
                    loss_time = time.time() - loss_begin
                                         
                    eval_begin = time.time() 
                    hits, ndcgs, losses = [],[],[]
                    for test_data in self.gd.generateTestData():
                        user_input_data = np.array(test_data[0]).astype(np.int32)

#                        time_input_data = np.array(test_data[1]).astype(np.float32)

                        item_input_data= np.array(test_data[1]).astype(np.int32)
                        
                        labels_data = np.array(test_data[2]).astype(np.float32)
                        
                        feed_dict = {self.user_idx: user_input_data,  self.item_idx: item_input_data, self.labels: labels_data}
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

                                
    
        
    
