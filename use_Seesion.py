# -*- coding: utf-8 -*-
# @Time    : 2018/10/3 9:41
# @Author  : yc1307
# @File    : use_Seesion.py
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义两个矩阵
matrix1 = tf.constant([[2, 3]])
matrix2 = tf.constant([[3], [2]])

product = tf.matmul(matrix1, matrix2)  # matrix multiple  矩阵乘法

"""
# 直接创建并执行Session会话
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
"""

# 使用with语句执行Session会话
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
