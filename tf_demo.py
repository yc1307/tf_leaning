# -*- coding: utf-8 -*-
# @Time    : 2018/8/17 11:47
# @Author  : XiZhi
# @File    : tf_demo.py
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
