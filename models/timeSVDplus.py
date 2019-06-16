# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import math

class timeSVD(object):
    
    def __init__(self, config, dl, gd):
        self.config = config
        self.dl = dl
        self.gd = gd
        self.num_items = self.dl.num_items
        self.num_users = self.dl.num_users
        self.factors = self.config.factors
        self.numT = self.dl.numT
        self.regU1 = self.config.regU1
        self.regU2 = self.config.regU2
        self.regU3 = self.config.regU3
        self.regI = self.config.regI

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_idx = tf.placeholder(tf.int32, shape=[None,], name='user_idx')
            self.item_idx = tf.placeholder(tf.int32, shape=[None,], name = 'item_idx')
            self.ut_idx = tf.placeholder(tf.int32, shape=[None, 2], name='ut_idx') # userid, t
            self.labels = tf.placeholder(tf.float32, shape=[None,1], name='labels')
            self.tu = tf.placeholder(tf.float32, shape=[None,1], name='tu')
  
    
    def _create_variables(self):
        with tf.name_scope('embeddings'):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.factors],mean=0.0,stddev=0.01), \
                                            dtype=tf.float32, name='embedding_P')
            self.embedding_A = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.factors],mean=0.0,stddev=0.01), \
                                           dtype=tf.float32, name='embedding_A')
            self.embedding_PT = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.numT, self.factors],mean=0.0,stddev=0.01), \
                                            dtype=tf.float32, name='embedding_PT')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.factors],mean=0.0,stddev=0.01), \
                                            dtype=tf.float32, name='embedding_Q')
            
    
    def _create_inference(self):
        with tf.name_scope('inference'):
            self.embedding_pu = tf.nn.embedding_lookup(self.embedding_P, self.user_idx) #(b,k)
            self.embedding_au = tf.nn.embedding_lookup(self.embedding_A, self.user_idx) #(b,k)
            self.embedding_put = tf.gather_nd(self.embedding_PT, self.ut_idx) #(b,,k)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_idx)
            self.pu = self.embedding_pu  + self.embedding_put + self.embedding_au*self.tu
            self.output = tf.sigmoid(tf.expand_dims(tf.reduce_sum(self.pu * self.embedding_q,1),1))
    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) +  \
                                          self.regU1*tf.reduce_sum(tf.square(self.embedding_P)) + \
                                        self.regI*tf.reduce_sum(tf.square(self.embedding_Q)) +self.regU2*tf.reduce_sum(tf.square(self.embedding_A))  \
                                        +self.regU3*tf.reduce_sum(tf.square(self.embedding_PT)) 
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
                for train_data in self.gd.generateNormalTrainData():
                    
                    user_input_data = np.array(train_data[0]).astype(np.int32)
                    ut_input_data = np.array(train_data[1]).astype(np.float32)
                    item_input_data= np.array(train_data[2]).astype(np.int32)
                    
                    labels_data = np.array(train_data[3]).astype(np.float32)
                    time_input_data = np.array(train_data[4]).astype(np.int32)
                    tu_data = time_input_data - train_data[5]
                    feed_dict = {self.user_idx: user_input_data,  self.item_idx: item_input_data,self.ut_idx:ut_input_data, self.labels: labels_data[:, np.newaxis], self.tu:tu_data[:, np.newaxis]}
                    training_loss, _ = sess.run([self.loss, self.optimizer], feed_dict)
                train_time = time.time() - train_begin
                if epoch_count % self.config.verbose_count ==0:
                    #train loss
                    loss_begin = time.time()
                    train_loss = 0.0
                    batch_i = 0
                    for train_data in self.gd.generateNormalTrainData():
                    
                        user_input_data = np.array(train_data[0]).astype(np.int32)
                        ut_input_data = np.array(train_data[1]).astype(np.float32)
                        item_input_data= np.array(train_data[2]).astype(np.int32)
                        
                        labels_data = np.array(train_data[3]).astype(np.float32)
                        time_input_data = np.array(train_data[4]).astype(np.int32)
                        tu_data = time_input_data - train_data[5]
                        feed_dict = {self.user_idx: user_input_data,  self.item_idx: item_input_data,self.ut_idx:ut_input_data, self.labels: labels_data[:, np.newaxis],  self.tu:tu_data[:, np.newaxis]}
                        train_loss += sess.run(self.loss, feed_dict)
                        batch_i+=1
                    train_loss = train_loss/batch_i
                    loss_time = time.time() - loss_begin
                                         
                    eval_begin = time.time() 
                    hits, ndcgs, losses = [],[],[]
                    for test_data in self.gd.generateNormalTestData():
                        print('#######')
                        user_input_data = np.array(test_data[0]).astype(np.int32)

                        time_input_data = np.array(test_data[1]).astype(np.float32)

                        item_input_data= np.array(test_data[2]).astype(np.int32)
                        
                        labels_data = np.array(test_data[3]).astype(np.float32)
                        
                        time_input_data = np.array(test_data[4]).astype(np.int32)
                        tu_data =  time_input_data - test_data[5]
                        feed_dict = {self.user_idx: user_input_data,  self.item_idx: item_input_data,self.ut_idx:ut_input_data, self.labels: labels_data[:, np.newaxis],  self.tu:tu_data[:, np.newaxis]}
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

                                
    
        
    