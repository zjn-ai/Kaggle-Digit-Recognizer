# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:12:35 2018

@author: zjn
"""
import tensorflow as tf
import pandas as pd 
import numpy as np
test = pd.read_csv('test.csv')

X_test = test.iloc[:,:].values
X_test = np.float32(X_test/255.)
X_test = np.reshape(X_test, [28000,28,28,1])


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model\model.ckpt.meta')
    saver.restore(sess, "model\model.ckpt")
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    training = graph.get_tensor_by_name("training:0")
    y = graph.get_tensor_by_name("y:0")
    for i in range(7):
        prediect = sess.run(y, feed_dict={X:X_test[i*4000:(i+1)*4000,:,:,:], training:0})
        try:
            result = np.hstack((result,prediect))
        except:
            result = prediect
    pd.DataFrame({"ImageId": range(1, len(result) + 1), "Label": result}).to_csv('submission.csv', index=False, header=True)