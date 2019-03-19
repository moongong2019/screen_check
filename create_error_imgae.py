#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：moon time:2019/3/5
import cv2 as cv
import os
import random
import numpy as np


class CreateErrorImage(object):
    def __init__(self,image_dir,error_image_dir):
        self._img_dir =image_dir
        self._error_image_dir = error_image_dir
        if not os.path.exists(self._error_image_dir):
            os.mkdir(self._error_image_dir)

    def get_image_list(self):
        if os.path.isdir(self._img_dir):
            return list(filter(lambda y: '.jpg'or '.png' in y, os.listdir(self._img_dir)))
        else:
            print('not a dir')
            return []

    #在一张正常图片中生成count数量的黑块
    def create_black_rect(self,img_path,count):
        img = cv.imread(img_path)
        height, weight = img.shape[:2]
        for i in range(count):
            height_start = random.randint(0, height - 50)
            height_end = height if height_start + random.randint(50, 200) > height else height_start + random.randint(50,200)
            weight_start = random.randint(0, weight - 50)
            weight_end = weight if weight_start + random.randint(50, 200) > weight else weight_start + random.randint(50,200)
            # 将区域置为黑色
            img[height_start:height_end, weight_start:weight_end] = 0
        error_image_path = os.path.join(os.getcwd(),self._error_image_dir,"error_blackRect_"+img_path.split('\\')[-1])
        #print(error_image_path)
        cv.imwrite(error_image_path,img)
        return error_image_path

    # 在一个图片中随机生成count个白色色块
    def create_white_rect(self,img_path,count):
        img = cv.imread(img_path)
        height, weight = img.shape[:2]
        for i in range(count):
            height_start = random.randint(0, height - 50)
            height_end = height if height_start + random.randint(50, 200) > height else height_start + random.randint(50,200)
            weight_start = random.randint(0, weight - 50)
            weight_end = weight if weight_start + random.randint(50, 200) > weight else weight_start + random.randint(50,200)
            # 将区域置为白色
            img[height_start:height_end, weight_start:weight_end] = 255

        error_image_path = os.path.join(os.getcwd(), self._error_image_dir, "error_whiteRect_" + img_path.split('\\')[-1])
        # print(error_image_path)
        cv.imwrite(error_image_path, img)
        return error_image_path

    # 在一个图片中随机生成count个随机色块
    def create_rndcolor_rect(self,img_path,count):
        img = cv.imread(img_path)
        height, weight = img.shape[:2]
        for i in range(count):
            height_start = random.randint(0, height - 50)
            height_end = height if height_start + random.randint(50, 200) > height else height_start + random.randint(50,200)
            weight_start = random.randint(0, weight - 50)
            weight_end = weight if weight_start + random.randint(50, 200) > weight else weight_start + random.randint(50,200)
            # 将区域置为随机色
            img[height_start:height_end, weight_start:weight_end] = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

        error_image_path = os.path.join(os.getcwd(), self._error_image_dir, "error_rndcolorRect_" + img_path.split('\\')[-1])
        # print(error_image_path)
        cv.imwrite(error_image_path, img)
        return error_image_path

    # 在一个图片中随机生成count条横线
    def create_rndcolor_hengxian(self,img_path,count):
        img = cv.imread(img_path)
        height, weight = img.shape[:2]
        for i in range(count):
            height_start = random.randint(0, height - 2)
            height_end = height if height_start + 1 > height else height_start + 1
            img[height_start:height_end, :] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        error_image_path = os.path.join(os.getcwd(), self._error_image_dir, "error_hengxian" + img_path.split('\\')[-1])
        # print(error_image_path)
        cv.imwrite(error_image_path, img)
        return error_image_path

    # 在一个图片中随机生成count条竖线
    def create_rndcolor_shuxian(self,img_path,count):
        img = cv.imread(img_path)
        height, weight = img.shape[:2]
        for i in range(count):
            weight_start = random.randint(0, weight - 2)
            weight_end = weight if weight_start + 1 > weight else weight_start + 1
            img[:, weight_start:weight_end] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        error_image_path = os.path.join(os.getcwd(), self._error_image_dir, "error_shuxian" + img_path.split('\\')[-1])
        # print(error_image_path)
        cv.imwrite(error_image_path, img)
        return error_image_path

    # 将图片平移，从而产生拉升效果，横向平移时y坐标传0
    def translate(self, img,translate_x, translate_y):
        src_h, src_w = img.shape[:2]
        dst = np.zeros(img.shape, dtype=np.uint8)
        for row in range(src_h):
            for col in range(src_w):
                h = int(row - translate_y)
                w = int(col - translate_x)
                if h < src_h and h >= 0 and w < src_w and w >= 0:
                    dst[row][col] = img[h][w]
        dst[:, 0:translate_x] = img[:, 0:translate_x]
        dst[0:translate_y, :] = img[0:translate_y, :]
        return dst

    # 将图片横向拉升N次
    def hengxiang_translate(self,img_path,translate_x,count):
        print("translet imge %s"%image_path)
        img = cv.imread(img_path)
        for i in range(count):
            img = self.translate(img,translate_x,0)
            translate_x = translate_x*count

        error_image_path = os.path.join(os.getcwd(), self._error_image_dir, "error_hxTrans" + img_path.split('\\')[-1])
        # print(error_image_path)
        cv.imwrite(error_image_path, img)
        return error_image_path

    # 将图片竖向拉升N次
    def shuxiang_translate(self, img_path, translate_y, count):
        print("translet imge %s" % image_path)
        img = cv.imread(img_path)
        for i in range(count):
            img = self.translate(img, 0,translate_y)
            translate_y = translate_y * count

        error_image_path = os.path.join(os.getcwd(), self._error_image_dir, "error_zxTrans" + img_path.split('\\')[-1])
        # print(error_image_path)
        cv.imwrite(error_image_path, img)
        return error_image_path


def create_rect_image(image_path,error_image_path):
    error_imgae = CreateErrorImage(image_path, error_image_path)
    img_list = error_imgae.get_image_list()
    print('create error image')
    for idx, image in enumerate(img_list):
        path = os.path.join(os.getcwd(), image_path, image)
        if idx % 3 == 0:
            error_imgae.create_black_rect(path, random.randint(1, 4))
        elif idx % 3 == 1:
            error_imgae.create_white_rect(path, random.randint(1, 4))
        else:
            error_imgae.create_rndcolor_rect(path, random.randint(1, 4))
    print('create error image done')


def create_xiantiao_image(image_path,error_image_path):
    error_imgae = CreateErrorImage(image_path, error_image_path)
    img_list = error_imgae.get_image_list()
    print('create error image')
    for idx, image in enumerate(img_list):
        path = os.path.join(os.getcwd(), image_path, image)
        if idx % 2 == 0:
            error_imgae.create_rndcolor_shuxian(path,random.randint(1,20))
        else:
            error_imgae.create_rndcolor_hengxian(path,random.randint(1,20))
    print('create error image done')


def create_chongdie_image(image_path,error_image_path):
    error_imgae = CreateErrorImage(image_path, error_image_path)
    img_list = error_imgae.get_image_list()
    print('create error image')
    for idx, image in enumerate(img_list):
        path = os.path.join(os.getcwd(), image_path, image)
        if idx % 2 == 0:
            error_imgae.hengxiang_translate(path,8,random.randint(3,8))
        else:
            error_imgae.shuxiang_translate(path,8,random.randint(3,8))
    print('create error image done')

if __name__ == '__main__':
    image_path = 'videoToImage'
    error_image_path = 'error_chongdie_image'
    create_chongdie_image(image_path, error_image_path)
