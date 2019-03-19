#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：moon time:2019/3/5
import cv2
import os,time


class GetImageFromVideo(object):
    def __init__(self, video_dir,image_dir):
        self._video_dir = video_dir
        self._image_dir = image_dir
        if not os.path.exists(self._image_dir):
            os.mkdir(self._image_dir)

    def get_video_list(self):
        if os.path.isdir(self._video_dir):
            return list(filter(lambda y: '.mp4'  in y, os.listdir(self._video_dir)))
        else:
            print('not a dir')
            return []

    def read_vedio_kernel(self,vedio_path):
        videoCapture = cv2.VideoCapture(vedio_path)

        # 获得码率及尺寸
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps, size)
        #fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        #print(fNUMS)
        num = 0
        start = time.time()
        count = 0
        while True:
            # start = time.time()
            ret, frame = videoCapture.read()
            if frame is None:
                break
            end = time.time()
            seconds = end - start
            num = num + 1
            nfps = num / seconds
            print("nfps:%d" % nfps)
            cv2.namedWindow("video")
            cv2.imshow('video', frame)
            if num % 15 ==0:
                filename = os.path.join(os.getcwd(),self._image_dir,vedio_path.split('\\')[-1][:-4]+"_"+str(count) + '.jpg')
                #print(filename)
                count+=1
                cv2.imwrite(filename, frame)
            c = cv2.waitKey(delay=int(fps))

        videoCapture.release()
        cv2.destroyAllWindows()

def main():
    video_dir = 'videos'
    image_dir = 'videoToImage'
    readVideo = GetImageFromVideo(video_dir, image_dir)
    video_list = readVideo.get_video_list()
    print('get image from video....')
    for video in video_list:
        path = os.path.join(os.getcwd(), video_dir, video)
        readVideo.read_vedio_kernel(path)

    print('done ...')
if __name__ =="__main__":
    #main()
    img = cv2.imread('test.jpg')
    cv2.resize(img,84,84)
    cv2.namedWindow("test")
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()