#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：moon time:2019/3/6
import tensorflow as tf
import numpy as np
import os
import cv2 as cv

##将所有的图片resize成100*100
H = 100
W = 100
C = 3   #RGB 三个通道


def get_train_data(img_path):
    img_dir_list = [img_path+x for x in os.listdir(img_path) if os.path.isdir(img_path+x)]
    #print(img_dir_list)
    img_list = []
    label_list = []
    for idx, folder in enumerate(img_dir_list):
        if 'error_rect' in folder:
            print("read image from folder:%s"%folder)
            for img in os.listdir(folder):
                if '.jpg' in img:
                    print(img)
                    im = cv.imread(os.path.join(folder,img))
                    im = cv.resize(im,(W,H))
                    img_list.append(im)
                    label_list.append(np.array([0,0,0,1])) #类别1
            print("done from %s"%folder)
        elif 'error_xiantiao'in folder:
            print("read image from folder:%s" % folder)
            for img in os.listdir(folder):
                if '.jpg' in img:
                    print(img)
                    im = cv.imread(os.path.join(folder,img))
                    im = cv.resize(im,(W,H))
                    img_list.append(im)
                    label_list.append(np.array([0,0,1,0]))#类别2
            print("done from %s" % folder)
        elif 'error_chongdie' in folder:
            print("read image from folder:%s" % folder)
            for img in os.listdir(folder):
                if '.jpg' in img:
                    print(img)
                    im = cv.imread(os.path.join(folder, img))
                    im = cv.resize(im, (W, H))
                    img_list.append(im)
                    label_list.append(np.array([0, 1, 0, 0]))#类别3
            print("done from %s" % folder)
        elif 'normal' in folder:
            print("read image from folder:%s" % folder)
            for img in os.listdir(folder):
                if '.jpg' in img:
                    print(img)
                    im = cv.imread(os.path.join(folder, img))
                    im = cv.resize(im, (W, H))
                    img_list.append(im)
                    label_list.append(np.array([1, 0, 0, 0]))#类别4
            print("done from %s" % folder)
    print(len(img_list),len(label_list))

    img_list = np.asarray(img_list).astype(np.uint8)
    label_list = np.asarray(label_list)
    #打乱顺序存储
    num_example = img_list.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = img_list[arr]
    label = label_list [arr]
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save('data/whole_data.npy',data)
    np.save('data/whole_label.npy',label)

get_train_data('images/')
