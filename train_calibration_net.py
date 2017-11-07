'''
Implementation of "A Convolutional Neural Network Cascade for Face Detection "
Paper : https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
Author : Dennis Liu
Modify : 2017/11/04

Description :   The example of training calibration nets , 
                I change the size to 48*48 in 12_cal_net and 24_cal_net
                because of the loss alway not decrease by using original size.

'''
import numpy as np
import tensorflow as tf
    
import model


from data import DataSet
from dataset.fddb_crawler import parse_data_info


def train_cal_net():
    # get only the positive training sample
    data_info = parse_data_info(only_positive = True)


    # training configuration
    batch = 100
    size = (48,48,3)
    start_epoch = 0
    end_epoch = 1000
    train_validation_rate = 0.9 # training set / all sample

    # load the pretrained model , set None if you don't have
    pretrained =  'models/48_cal_net_5.ckpt'

    # load data iterater
    dataset = DataSet(data_info,train_rate = train_validation_rate)
    _ , train_op , val_op , next_ele = dataset.get_iterator(batch,size)

    
    # load network
    net_12_c = model.calib_12Net(lr = 0.0000005)
    net_24_c = model.calib_24Net(lr = 0.0000005)
    net_48_c = model.calib_48Net(lr = 0.0000005)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    if pretrained:
        saver.restore(sess , pretrained)
        
    else:
        sess.run(tf.global_variables_initializer())    
    
    

    for epoch in xrange(start_epoch,end_epoch):
        loss = 0
        iteration = 0
        sess.run(train_op)
        # get each element of the training dataset until the end is reached
        while True:
            try:
                # default of the size returned from data iterator is 48
                inputs_48,clss ,pattern = sess.run(next_ele)
                # <ndarray> , <0/1> , <one-hot of 45-class>


                clss = clss.reshape(batch,1)
                pattern = pattern.reshape(batch,45)
                

                # resize image into shape of 12,24
                inputs_12 = inputs_48.copy()
                inputs_12.resize(batch,12,12,3)
                inputs_24 = inputs_48.copy()
                inputs_24.resize(batch,24,24,3)

                '''Put the size(48,48) into 12_cal_net and 24_cal_net ,because of the origrinal size is too small to convergence'''
                train_nets = [net_12_c,net_24_c,net_48_c]
                net_feed_dict = {net_12_c.inputs:inputs_48 , net_12_c.targets:pattern,\
                                net_24_c.inputs:inputs_48 , net_24_c.targets:pattern,\
                                net_48_c.inputs:inputs_48 , net_48_c.targets:pattern,}

                # training net
                sess.run([net.train_step for net in train_nets],\
                        feed_dict = net_feed_dict)
                # loss computation
                losses = sess.run([net.loss for net in train_nets],\
                        feed_dict = net_feed_dict)

                if iteration % 100 == 0:
                    net_12_c_eva = net_12_c.evaluate(inputs_48,pattern)
                    net_12_c_acc = sum(net_12_c_eva)/len(net_12_c_eva)
                    net_24_c_eva = net_24_c.evaluate(inputs_48,pattern)
                    net_24_c_acc = sum(net_24_c_eva)/len(net_24_c_eva)
                    net_48_c_eva = net_48_c.evaluate(inputs_48,pattern)
                    net_48_c_acc = sum(net_48_c_eva)/len(net_48_c_eva)
                    print ('Training Epoch {} --- Iter {} --- Training Accuracy:  {}%,{}%,{}% --- Training Loss: {}'\
                            .format(epoch , iteration , net_12_c_acc , net_24_c_acc , net_48_c_acc  , losses))

                iteration += 1
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        
        # get each element of the validation dataset until the end is reached
        sess.run(val_op)
        net_12_c_acc = []
        net_24_c_acc = []
        net_48_c_acc = []
        while True:
            try:
                # the size returned from data iterator is 48
                inputs_48,clss ,pattern = sess.run(next_ele)
                clss = clss.reshape(batch,1)
                pattern = pattern.reshape(batch,45)

                #resize image into shape of 12,24,48
                inputs_12 = inputs_48.copy()
                inputs_12.resize(batch,12,12,3)
                inputs_24 = inputs_48.copy()
                inputs_24.resize(batch,24,24,3)

                
                net_12_c_eva = net_12_c.evaluate(inputs_48,pattern)
                net_24_c_eva = net_24_c.evaluate(inputs_48,pattern)
                net_48_c_eva = net_48_c.evaluate(inputs_48,pattern)
                for i in range(len(net_12_c_eva)):
                    net_12_c_acc.append(net_12_c_eva[i])
                    net_24_c_acc.append(net_24_c_eva[i])
                    net_48_c_acc.append(net_48_c_eva[i])
            except tf.errors.OutOfRangeError:
                print("End of validation dataset.")
                break

        print ('Validation Epoch {}  Validation Accuracy:  {}%,{}%,{}%'\
                            .format(epoch , sum(net_12_c_acc)/len(net_12_c_acc),\
                                            sum(net_24_c_acc)/len(net_24_c_acc),\
                                            sum(net_48_c_acc)/len(net_48_c_acc)))

        saver = tf.train.Saver()
        save_path = saver.save(sess, "models/48_cal_net_{}.ckpt".format(epoch))
        print ("Model saved in file: ".format(save_path)) 

if __name__ == "__main__":
    train_cal_net()