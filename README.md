# Kaggle-Digit-Recognizer
> 本人原创，转载请注明出处https://blog.csdn.net/zjn295771349/article/details/84330967


 

贴出来我的比赛结果,截至到现在是TOP 15%，用的是CNN，完整的代码我会贴在最后面。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181121192903897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pqbjI5NTc3MTM0OQ==,size_16,color_FFFFFF,t_70)


# 洗数据

 1. 从kaggle下载的数据集包含三个文件，train.csv，test.csv和sample_submission.csv。
 2. 利用pandas包读取.csv文件。
 3. train.csv是42000x785的数组，一共42000个样本，第一列是图像的label，剩下784需要转换为28x28的图片。
 4. test.csv是28000x784的数组，网络训练好之后对其识别，结果放到sample_submission.csv中上传到kaggle评估。
 5. 将train中的前40000个样本拿出来作为训练集，剩下2000个样本作为交叉验证集。网络模型调好之后可以把42000个样本数据都放入训练集（本人懒，没放），放入之后可能会稍微提高一点正确率。

```python
train = pd.read_csv('train.csv')
X_train = train.iloc[:40000,1:].values
X_train = np.float32(X_train/255.)#将输入值区间调整到[0,1]
X_train = np.reshape(X_train, [40000,28,28,1])#将行向量转换为图片，因为用的是CNN
Y_train = train.iloc[:40000,0].values
X_dev = train.iloc[40000:,1:].values
X_dev = np.float32(X_dev/255.)
X_dev = np.reshape(X_dev, [2000,28,28,1])
Y_dev = train.iloc[40000:,0].values
```
## ONE - HOT

 1. 因为是多分类需要将label进行编码。

```python
Y_train = np.eye(10)[Y_train.reshape(-1)]
Y_dev = np.eye(10)[Y_dev.reshape(-1)]
```
# CNN结构

 1. 用了三个卷积层，每个卷积层后面都跟了relu激活函数，最大池化层和dropout，最后是两个全连接层。
   
 2. dropout层的设置参考了kaggle上的[Yassine Ghouzam, PhD大神的markdown](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)，不得不说防止过拟合的效果真的很好。
 3. 在张量中设置name，是为了以后保存网络和读取网络进行测试时能够找到图的输入和输出。
 4. 在调参的时候无意间发现当第二个池化层的步长为（1，1）的时候，拟合的速度很快，而且准确率很高。
 5. 卷积核我用的都是3x3的，试过第一卷积层改为5x5的，差别不大。

```python
#设置3个placeholder，是喂数据的入口
X = tf.placeholder(tf.float32,(None,28,28,1), name = 'X') 
Y = tf.placeholder(tf.float32,(None,10))
training = tf.placeholder(tf.bool, name = 'training')#这个placeholder是为了在训练的时候开启dropout，测试的时候关闭dropout
W1 = tf.get_variable('W1',[3,3,1,32],initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())

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
```
# 设置GPU

 1. po主用的是1050ti，如果不设置，tensorflow会占满显存，后续训练的时候会爆显存的，没有的小伙伴可以不用设置。

```python
gpu_no = '0' # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
# 定义TensorFlow配置
config = tf.ConfigProto()
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
```
# 学习率自适应

 1. 当快接近全局最优点时，如果学习率很大会在最优点附近震荡，所以随之迭代的次数增多需要减小学习率。

```python
starter_learning_rate = 1e-4
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1500, 0.96, staircase = True)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)
init = tf.global_variables_initializer()
```
# 制作mini batch

 1. mini batch会让训练的速度快很多。

```python
def random_mini_batches(X, Y, mini_batch_size):
    
    m = X.shape[0]                  
    mini_batches = []
    
    # 打乱数据
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # 制作mini batch
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # 如果样本数不能被mini batch的size整除，则将剩下的整合起来
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
```
# 开始训练及保存网络

 1. 因为显卡的显存有限，测试的时候将所有数据一次喂进去的时候会爆显存，所以测试训练集准确率的时候只用了前4000个样本。
 2. 当交叉验证集的正确率大于等于0.997或者迭代1000次，保存网络。迭代1000次1050ti大概用了1个小时左右，其实迭代3/400次就差不多了。

```python
with tf.Session() as sess:
    sess.run(init)#初始化参数
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
            print(sess.run(learning_rate))#为了观察学习率
            if Dev_Accuracy >=0.997 or epoch == 1000:
                saver.save(sess,'Model/model.ckpt')#保存网络
                break
        if epoch % 1 == 0:
            costs.append(minibatch_cost)#记录cost
```
# 加载网络并制作submission.csv

 1. 因为显存不足，所以分批将测试集喂到网络中。
 2. 最后提交结果到kaggle的时候需要vpn，不然上传不成功。
 3. 制作.csv文件参考了[这篇博客](https://blog.csdn.net/neruda1991/article/details/78745676)。

```python
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
```
# 完整代码

```python
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


```

