# -*- coding: utf-8 -*-
# @Time    : 2018/10/2 19:18
# @Author  : yc1307
# @File    : first_demo.py
import tensorflow as tf
import numpy as np
import os
# 设置 TensorFlow 的 Log 输出级别
# 1 默认等级，显示所有信息
# 2 只显示 warning 和 Error
# 3 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


x_data = np.random.rand(100).astype(np.float)
# 返回100个取值范围是[0,1)均匀分布的随机样本值。 astype：转换数组的数据类型。
y_data = x_data * 0.1 + 0.3

# create TensorFlow structure
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# random_uniform 均匀分布[min,max)
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# instantiation
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))

