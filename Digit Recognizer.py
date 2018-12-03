# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:54:05 2018

@author: zjn
"""
import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd

train = pd.read_csv('train.csv')

X_train = train.iloc[:40000,1:].values
X_train = np.float32(X_train/255.)
X_train = np.reshape(X_train, [40000,28,28,1])
Y_train = train.iloc[:40000,0].values
X_dev = train.iloc[40000:,1:].values
X_dev = np.float32(X_dev/255.)
X_dev = np.reshape(X_dev, [2000,28,28,1])
Y_dev = train.iloc[40000:,0].values

Y_train = np.eye(10)[Y_train.reshape(-1)]
Y_dev = np.eye(10)[Y_dev.reshape(-1)]

gpu_no = '0' # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
# 定义TensorFlow配置
config = tf.ConfigProto()
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

num_epochs = 6000
minibatch_size = 128
costs = []
m = X_dev.shape[0]

X = tf.placeholder(tf.float32,(None,28,28,1), name = 'X') 
Y = tf.placeholder(tf.float32,(None,10))
training = tf.placeholder(tf.bool, name = 'training')

W1 = tf.get_variable('W1',[3,3,1,32],initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable('W4',[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer())

conv_0 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
act_0 = tf.nn.relu(conv_0)
pool_0 = tf.nn.max_pool(act_0, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
dropout_0 = tf.contrib.layers.dropout(pool_0, keep_prob = 0.25, is_training = training)

conv_1 = tf.nn.conv2d(pool_0, W2, strides = [1,1,1,1], padding = 'VALID')
act_1 = tf.nn.relu(conv_1)
pool_1 = tf.nn.max_pool(act_1, ksize = [1,2,2,1],strides = [1,1,1,1], padding = 'VALID')
dropout_1 = tf.contrib.layers.dropout(pool_1, keep_prob = 0.25, is_training = training)

conv_2 = tf.nn.conv2d(pool_1, W3, strides = [1,1,1,1], padding = 'VALID')
act_2 = tf.nn.relu(conv_2)
pool_2 = tf.nn.max_pool(act_2, ksize = [1,2,2,1],strides = [1,1,1,1], padding = 'VALID')
dropout_2 = tf.contrib.layers.dropout(pool_2, keep_prob = 0.25, is_training = training)

flat = tf.contrib.layers.flatten(dropout_2)
full_0 = tf.contrib.layers.fully_connected(flat, 256, activation_fn = tf.nn.relu)
dropout_3 = tf.contrib.layers.dropout(full_0, keep_prob = 0.5, is_training = training)

full_1 = tf.contrib.layers.fully_connected(dropout_3, 10, activation_fn= None)

y = tf.argmax(full_1, 1, name = 'y')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=full_1,labels=Y))

def random_mini_batches(X, Y, mini_batch_size):
    
    m = X.shape[0]                  
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

starter_learning_rate = 1e-4
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1500, 0.96, staircase = True)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y,training:1})
            minibatch_cost += temp_cost / num_minibatches
        if epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            correct_prediction = tf.equal(tf.argmax(full_1,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))          
            print ("Train Accuracy:", accuracy.eval({X: X_train[:4000,:,:,:], Y: Y_train[:4000,:], training:0}))
            Dev_Accuracy = accuracy.eval({X: X_dev, Y: Y_dev, training:0})
            print ("Dev Accuracy:", Dev_Accuracy)
            print(sess.run(global_step))
            print(sess.run(learning_rate))
            if Dev_Accuracy >=0.997 or epoch == 1000:
                saver.save(sess,'Model/model.ckpt')
                break
        if epoch % 1 == 0:
            costs.append(minibatch_cost)

