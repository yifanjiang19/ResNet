# -*- coding: utf-8 -*-
from cifar_read import *

import tensorflow as tf
from tensorflow.python import debug as tf_debug
learning_rate = 0.0001
batch_size = 128

x = tf.placeholder("float32", shape=[None, 32*32*3])
labels = tf.placeholder("float32", shape=[None, 100])
keep_prob = tf.placeholder("float32")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, filter_size,input_num,output_num,b,strides,name=None):
    conv = tf.nn.conv2d(x,
                     weight_variable([filter_size,filter_size,input_num,output_num]),
                     strides=[1, strides, strides, 1],
                     padding="SAME",
                     name=name)
    x = tf.nn.bias_add(conv,bias_variable(b))
    return x

def max_pool(x,ksize,strides,name):
    output = tf.nn.max_pool(x,
                   ksize=[1,ksize,ksize,1],
                   strides=[1,strides,strides,1],
                   padding="SAME",
                   name=name)
    return tf.nn.relu(output)
# def res_add(input_layer,add_layer,dim_increase=None):
#     input_channel = input_layer.get_shape().as_list()[-1]
#     shortcut = tf.identity(input_layer)
#     if dim_increase == True:
#         pool_input = tf.nn.avg_pool(add_layer,ksize=[1,2,2,1],
#                                     strides=[1,1,1,1],padding="SAME")
#         padded_payer = tf.pad(pool_input,[[0,0],[0,0],[0,0],[input_channel//2,0]])
#         output = shortcut + padded_payer
#     else:
#         output = shortcut + add_layer
#     output = tf.nn.relu(output)
#     output = batchnormalize(output)
#     return output

def res_block(input,filter_size,input_dim,output_dim,dim_increase=False):
    input_channel = input.get_shape().as_list()[-1]
    shortcut = tf.identity(input)
    strides = 1
    if dim_increase == True:
        pool_input = tf.nn.avg_pool(input,ksize=[1,2,2,1],
                                    strides=[1,2,2,1],padding="SAME")
        padded_layer = tf.pad(pool_input,[[0,0],[0,0],[0,0],[input_channel//2,input_channel//2]])
        shortcut = padded_layer
        strides = 2
    conv_1 = conv2d(input,filter_size,input_dim,output_dim,[output_dim],strides)
    relu_1 = tf.nn.relu(conv_1)
    conv_2 = conv2d(relu_1,filter_size,output_dim,output_dim,[output_dim],1)
    add_layer = conv_2 + shortcut
    bn = tf.nn.relu(add_layer)
    output = batchnormalize(bn)
    return output


def fc_layer(x,input_size,output_size,name):
    w = weight_variable([input_size,output_size])
    b = bias_variable([output_size])
    output = tf.matmul(x,w)
    output = tf.add(output,b)
    return tf.nn.relu(output)

def batchnormalize(x):
    num = x.get_shape().as_list()[-1]
    output = tf.contrib.layers.batch_norm(x,
                      decay=0.9,
                      updates_collections=None,
                      epsilon=1e-5,
                      scale=True,
                      is_training=True,
                      )
    return tf.nn.relu(output)


x_image = tf.reshape(x,[-1,32,32,3])
print(x_image)
conv_1 = conv2d(x_image,7,3,64,[64],2)
print(conv_1)
pool_1 = max_pool(conv_1,3,2,name='pool_1')
print(pool_1)

#64
res_2 = res_block(pool_1,3,64,64)
res_3 = res_block(res_2,3,64,64)
res_4 = res_block(res_3,3,64,64)
bn_1 = batchnormalize(res_4)

#128
res_5 = res_block(bn_1,3,64,128,dim_increase=True)
res_6 = res_block(res_5,3,128,128)
res_7 = res_block(res_6,3,128,128)
res_8 = res_block(res_7,3,128,128)
bn_2 = batchnormalize(res_8)

#256
res_9 = res_block(bn_2,3,128,256,dim_increase=True)
res_10 = res_block(res_9,3,256,256)
res_11 = res_block(res_10,3,256,256)
res_12 = res_block(res_11,3,256,256)
res_13 = res_block(res_12,3,256,256)
res_14 = res_block(res_13,3,256,256)
bn_3 = batchnormalize(res_14)

#512
res_15 = res_block(bn_3,3,256,512,dim_increase=True)
res_16 = res_block(res_15,3,512,512)
res_17 = res_block(res_16,3,512,512)
bn_4 = batchnormalize(res_17)
# #64
# conv_2 = conv2d(pool_1,3,64,64,[64],1,'conv_2')
# conv_3 = conv2d(conv_2,3,64,64,[64],1,name='conv_3')
# conv_3 = res_add(conv_3,pool_1)
#
# conv_4 = conv2d(conv_3,3,64,64,[64],1,name='conv_4')
# conv_5 = conv2d(conv_4,3,64,64,[64],1,name='conv_5')
# conv_5 = res_add(conv_5,conv_3)
#
# conv_6 = conv2d(conv_5,3,64,64,[64],1,name='conv_6')
# conv_7 = conv2d(conv_6,3,64,64,[64],1,name='conv_7')
# conv_7 = res_add(conv_7,conv_5)
#
# conv_7 = batchnormalize(conv_7)
#
# #128
# conv_8 = conv2d(conv_7,3,64,128,[128],2,name='conv_8')
# conv_9 = conv2d(conv_8,3,128,128,[128],1,name='conv_9')
# conv_9 = res_add(conv_9,conv_7,dim_increase=True)
#
# conv_10 = conv2d(conv_9,3,128,128,[128],1,name='conv_10')
# conv_11 = conv2d(conv_10,3,128,128,[128],1,name='conv_11')
# conv_11 = res_add(conv_11,conv_9)
#
# conv_12 = conv2d(conv_11,3,128,128,[128],1,name='conv_12')
# conv_13 = conv2d(conv_12,3,128,128,[128],1,name='conv_13')
# conv_13 = res_add(conv_13,conv_11)
#
# conv_14 = conv2d(conv_13,3,128,128,[128],1,name='conv_14')
# conv_15 = conv2d(conv_14,3,128,128,[128],1,name='conv_15')
# conv_15 = res_add(conv_15,conv_13)
#
# conv_15 = batchnormalize(conv_15)
#
# #256
# conv_16 = conv2d(conv_15,3,128,256,[256],2,name='conv_16')
# conv_17 = conv2d(conv_16,3,256,256,[256],1,name='conv_17')
# conv_17 = res_add(conv_17,conv_15,dim_increase=True)
#
# conv_18 = conv2d(conv_17,3,256,256,[256],1,name='conv_18')
# conv_19 = conv2d(conv_18,3,256,256,[256],1,name='conv_19')
# conv_19 = res_add(conv_19,conv_17)
#
# conv_20 = conv2d(conv_19,3,256,256,[256],1,name='conv_20')
# conv_21 = conv2d(conv_20,3,256,256,[256],1,name='conv_21')
# conv_21 = res_add(conv_21,conv_19)
#
# conv_22 = conv2d(conv_21,3,256,256,[256],1,name='conv_22')
# conv_23 = conv2d(conv_22,3,256,256,[256],1,name='conv_23')
# conv_23 = res_add(conv_23,conv_21)
#
# conv_24 = conv2d(conv_23,3,256,256,[256],1,name='conv_24')
# conv_25 = conv2d(conv_24,3,256,256,[256],1,name='conv_25')
# conv_25 = res_add(conv_25,conv_23)
#
# conv_26 = conv2d(conv_25,3,256,256,[256],1,name='conv_26')
# conv_27 = conv2d(conv_26,3,256,256,[256],1,name='conv_27')
# conv_27 = res_add(conv_27,conv_25)

# conv_27 = batchnormalize(conv_27)
#
#
# #512
# conv_28 = conv2d(conv_27,3,256,512,[512],2,name='conv_28')
# conv_29 = conv2d(conv_28,3,512,512,[512],1,name='conv_29')
# conv_29 = res_add(conv_29,conv_27,dim_increase=True)
#
# conv_30 = conv2d(conv_29,3,512,512,[512],1,name='conv_30')
# conv_31 = conv2d(conv_30,3,512,512,[512],1,name='conv_31')
# conv_31 = res_add(conv_31,conv_29)
#
# conv_32 = conv2d(conv_31,3,512,512,[512],1,name='conv_32')
# conv_33 = conv2d(conv_32,3,512,512,[512],1,name='conv_33')
# conv_33 = res_add(conv_33,conv_31)

avg_pool = tf.nn.avg_pool(bn_4,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
avg_pool = tf.reshape(avg_pool,[-1,512])
#drop_1 = tf.nn.dropout(avg_pool,keep_prob)

fc_1 = fc_layer(avg_pool,512,1000,name='fc_1')
#drop_2 = tf.nn.dropout(fc_1,keep_prob)
w_1 = weight_variable([1000,100])
fc_2 = tf.matmul(fc_1,w_1)
output = fc_2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

cifar = CIFAR(batch_size)
x_train,y_train =  cifar.input_train()
x_test,y_test = cifar.input_test()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_index = 0
    for epoch in range(256):
        loss_mean = 0
        acc_mean = 0
        for i in range(50000//batch_size):
            yb_train = map(cifar.one_hot,y_train[i*batch_size:i*batch_size+batch_size])
            train_dict = {x:x_train[i*batch_size:i*batch_size+batch_size],
                         labels:yb_train,
                         keep_prob:0.9}

            sess.run(train_step,feed_dict=train_dict)

            loss,train_accuracy = sess.run([cross_entropy,accuracy],feed_dict=train_dict)
            loss_mean += loss
            acc_mean += train_accuracy

            if i%300 == 0:
                yb_test = map(cifar.one_hot,y_test[test_index*batch_size:test_index*batch_size+batch_size])
                test_dict =  {x:x_test[test_index*batch_size:test_index*batch_size+batch_size],
                              labels:yb_test,
                              keep_prob:0.9}
                test_index+=1
                if (test_index+1)*batch_size>10000:
                    test_index = 0
                print(sess.run([accuracy],feed_dict=test_dict))
        print (epoch,loss_mean/(i+1),acc_mean/(i+1))
