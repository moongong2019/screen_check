#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：moon time:2019/3/7
import tensorflow as tf
import NN
import numpy as np
import os,shutil
import matplotlib.pyplot as plt
import cv2 as cv

#输入图片大小为100*100*3
H = 100
W = 100
C = 3
#输出4个类别：色块、条纹、重叠、正常
OUTPUT_SIZE = 4
BATCH_SIZE = 128
EPOCH_NUM = 40
LEARNING_RATE = 0.0005

#测试集和训练集分割函数.percent%的数据作为验证集
def my_train_test_split(whole_data_path,whole_label_path,percent):
    whole_data = np.load(whole_data_path)
    whole_label = np.load(whole_label_path)
    s = np.int(whole_data.shape[0]*percent)
    train_data = whole_data[:s]
    test_data = whole_data[s:]
    train_label = whole_label[:s]
    test_label =  whole_label[s:]
    return train_data, test_data, train_label, test_label

def get_placeholder():
    images_ph = tf.placeholder(tf.float32, shape=(None, H, W, C))
    labels_ph = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))
    return images_ph, labels_ph


def train(whole_data_path,whole_label_path):
    train_data, test_data, train_label, test_label = my_train_test_split(whole_data_path, whole_label_path, 0.8)
    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)  # drop out 的丢弃值
    logits = NN.get_logits(images_ph, OUTPUT_SIZE, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_ph,logits=logits))
    #计算训练后的准确率
    correct_pre = tf.equal(tf.argmax(logits,axis=1),tf.argmax(labels_ph,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

    train_op = NN.get_train_op(loss,LEARNING_RATE)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(config=session_config) as  sess:
        sess.run(init_op)

        def train_batch(batch_data,batch_label):
            feed_dict = {
                images_ph:(np.asarray(batch_data)/255.0).astype(np.float32),
                labels_ph:np.asarray(batch_label),
                keep_prob:0.5
            }
            _,loss_value= sess.run([train_op,loss],feed_dict=feed_dict)
            return loss_value

        #训练一个批次的数据，取数据的时候需要随机从训练数据里面取
        def train_epoch():
            batch_data = []
            batch_label = []
            for i, index in enumerate(np.random.permutation(len(train_data))):
                batch_data.append(train_data[index])#随机取一个数据
                batch_label.append(train_label[index])

                if len(batch_data) == BATCH_SIZE: #取了128个数据后开始训练
                    # Train the model
                    loss_value = train_batch(batch_data, batch_label)
                    # Clear old batch
                    batch_data = []
                    batch_label = []

                if i != 0 and i % 500 == 0:
                    print("Step %d: loss %03f" % (i, loss_value))

            if len(batch_data) != 0:#train_data里面最后剩下的不足BATCH_SIZE长度的数据部分也训练下
                train_batch(batch_data, batch_label)

        mean_loss_list = []
        mean_acc_list = []
        for i in range(EPOCH_NUM):
            print('开始训练...')
            train_epoch()
            #保存每一个批次训练完后的模型
            if not os.path.exists('ckpt'):
                os.mkdir('ckpt')
            save_path = saver.save(sess, 'ckpt/model.screenCheck_epoch' + str(i) + '.ckpt')
            print("Model saved in path: %s" % save_path)
            print('-------在测试集上进行验证----------')
            pre_loss_list = []
            acc_list = []
            for id, one_test_data in enumerate(test_data):
                one_test_data = one_test_data[np.newaxis, :]
                # print(one_test_data.shape)
                feed_dict = {
                    images_ph: (one_test_data/255.0).astype(np.float64),
                    labels_ph: np.array(test_label[id][np.newaxis, :]).astype(np.int32),
                    keep_prob: 1.0,
                }
                test_loss, acc = sess.run([loss, accuracy], feed_dict=feed_dict)

                pre_loss_list.append(test_loss)
                acc_list.append(acc)

            mean_loss = np.mean(np.array(pre_loss_list))
            mean_acc = np.mean(np.array(acc_list))
            print('mean predict  loss: %f,mean acc:%f' % (mean_loss, mean_acc))
            mean_loss_list.append(mean_loss)
            mean_acc_list.append(mean_acc)
            step = i + 1
            if mean_loss_list[i] > mean_loss_list[i - 1] and mean_loss_list[i - 1] > mean_loss_list[i - 2]:  # 连续2次损失上升停止训练
                print('训练%d次后，出现过拟合，停止训练' % (i - 2))
                shutil.copyfile('ckpt/model.screenCheck_epoch' + str(i - 2) + '.ckpt' + '.data-00000-of-00001', 'ckpt/model.screenCheck_epoch.ckpt' + '.data-00000-of-00001')
                shutil.copyfile('ckpt/model.screenCheck_epoch' + str(i - 2) + '.ckpt' + '.index','ckpt/model.screenCheck_epoch.ckpt' + '.index')
                shutil.copyfile('ckpt/model.screenCheck_epoch' + str(i - 2) + '.ckpt' + '.meta','ckpt/model.screenCheck_epoch.ckpt' + '.meta')
                break

        saver.save(sess, 'ckpt/model.screenCheck.ckpt')
        print("各个批次训练后在测试集的平均准确率为 ：")
        print(mean_acc_list)

        plt.xlabel('epoch')
        plt.ylabel('loss and acc')
        label = ['mean loss', 'mean acc']
        plt.plot(range(step), mean_loss_list, 'r')
        plt.plot(range(step), mean_acc_list, 'g')
        plt.legend(label, loc=0, ncol=2)
        plt.show()
        plt.savefig('train.jpg')

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
            #print(check_result)
            if check_result ==3:
                print("image %s is rect error"%img_path)
            elif check_result == 2:
                print("image %s is hengxiang or shuxiang error" % img_path)
            elif check_result == 1:
                print("image %s is Translate error" % img_path)
            elif check_result == 0:
                print("image %s is namal image" % img_path)
            else:
                print("image %s is unknow type" % img_path)


if __name__ == "__main__":
    whole_data_path = 'data/whole_data.npy'
    whole_label_path = 'data/whole_label.npy'
    train(whole_data_path=whole_data_path,whole_label_path=whole_label_path)
    #image_dir = 'imageTest/'
    #preidct(image_dir)