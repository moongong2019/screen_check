#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：moon time:2019/3/11

import tensorflow as tf
import NN
import numpy as np
import os
import cv2 as cv

#输入图片大小为100*100*3
H = 100
W = 100
C = 3
#输出4个类别：色块、条纹、重叠、正常
OUTPUT_SIZE = 4

def get_placeholder():
    images_ph = tf.placeholder(tf.float32, shape=(None, H, W, C))
    labels_ph = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))
    return images_ph, labels_ph

def preidct(image_dir):
    img_list = [image_dir + x for x in os.listdir(image_dir) if '.jpg' or '.png' in x ]
    #对每一张图片进行预测
    tf.reset_default_graph()
    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)  # drop out 的丢弃值
    logits = NN.get_logits(images_ph, OUTPUT_SIZE, keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'ckpt/model.screenCheck.ckpt')
        print("Model restored.")

        for img_path  in img_list:
            img = cv.imread(img_path)
            img = cv.resize(img,(W,H))
            feed_dict = {
                images_ph:(np.asarray(img[np.newaxis, :])/255.0).astype(np.float64),
                keep_prob:1.0,
            }
            logits_value =sess.run(logits,feed_dict)
            check_result = np.argmax(logits_value)
            probability = np.amax(logits_value) / np.sum(logits_value) * 100
            #print(check_result)
            if check_result ==3:
                print("图片 %s 有色块错误,概率为 %.2f%%" % (img_path, probability))
            elif check_result == 2:
                print("图片 %s 有线条错误，概率为 %.2f%%" % (img_path, probability))
            elif check_result == 1:
                print("图片 %s 有重叠错误,概率为 %.2f%%" % (img_path, probability))
            elif check_result == 0:
                print("图片 %s 为正常图片概率为 %.2f%%" % (img_path, probability))
            else:
                print("图片 %s 无法检测出结果，请核实图片是否正确" % img_path)

if __name__ == "__main__":
    image_dir = 'imageTest/'
    preidct(image_dir)
