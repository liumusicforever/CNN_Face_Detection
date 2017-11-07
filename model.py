'''
Implementation of "A Convolutional Neural Network Cascade for Face Detection "
Paper : https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
Author : Dennis Liu
Modify : 2017/11/04

Description : The tensorflow structure of models in paper , 12net,24net,48net (include detection and calibration)

'''


import tensorflow as tf
import numpy as np



def weight_variable(shape,name=None,lr_type = 'conv'):
    # weight initial problem is very importand during training
    # use tf.random_normal convergence slower then truncated
    if lr_type == 'conv': 
        initial = tf.truncated_normal(shape, dtype="float32", stddev = 0.01)
        # initial = tf.random_normal(shape=shape, mean=0, stddev=0.001)
    else:
        x = np.sqrt(6. / (np.prod(np.array(shape[:-1])) + shape[-1]))
        initial = tf.random_uniform(shape, minval=-x,maxval=x)
        
    return tf.Variable(initial,name=name)

def bias_variable(shape,name=None):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W, stride, pad = 'SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=pad)

def max_pool(x, kernelSz, stride, pad = 'SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernelSz, kernelSz, 1], strides=[1, stride, stride, 1], padding=pad)

#12-net
class detect_12Net:
    def __init__(self,size = (48,48,3),lr = 0.001 , is_train = True):
        # data,label
        self.inputs = tf.placeholder("float",[None,size[0],size[1],size[2]])
        self.targets = tf.placeholder("float", [None,1])

        with tf.variable_scope("12det_"):
        
            #conv layer 1
            self.w_conv1 = weight_variable([3,3,size[2],16],"w_conv1")
            self.b_conv1 = bias_variable([16],"b_conv1")
            self.conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, 1) + self.b_conv1)
            
            
            #pooling layer 1
            self.pool1 =  max_pool(self.conv1, 3, 2)
            

            #fully conv layer 1
            self.w_fc1 = weight_variable([size[0]/2 * size[1]/2 * 16, 16],'w_fc1',lr_type = 'fc')
            self.b_fc1 = bias_variable([16],'b_fc1')
            self.pool1_flat = tf.reshape(self.pool1, [-1, size[0]/2 * size[1]/2 *16])
            self.fc1 = tf.nn.relu(tf.matmul(self.pool1_flat, self.w_fc1) + self.b_fc1)
            


            #fully conv layer 2
            self.w_fc2 = weight_variable([16, 1],'w_fc2',lr_type = 'fc')
            self.b_fc2 = bias_variable([1],'b_fc2')
            self.fc2 = tf.matmul(self.fc1,self.w_fc2) + self.b_fc2
        if is_train:
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2,labels =self.targets))
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)  
    
    def get_fc(self,inputs_12):
        return self.fc1.eval(feed_dict = {self.inputs:inputs_12})

    def evaluate(self,inputs_12,targets):
        predict = tf.to_float(tf.greater(tf.nn.sigmoid(self.fc2),0.5))
        label   =  targets
        eva = tf.cast(tf.equal(predict,label),"float").eval(feed_dict = {self.inputs:inputs_12, self.targets:targets})
        return eva

#24-net
class detect_24Net:
    def __init__(self,size = (48,48,3) ,lr = 0.001, is_train = True):

        # data,label
        self.inputs = tf.placeholder("float",[None,size[0],size[1],size[2]])
        self.targets = tf.placeholder("float", [None,1])
        # the fc1 from 12net
        self.from_12 = tf.placeholder("float",[None,16])

        with tf.variable_scope("24det_"):
            #conv layer 1
            self.w_conv1 = weight_variable([3,3,size[2],64],"w_conv1")
            self.b_conv1 = bias_variable([64],"b_conv1")
            self.conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, 1) + self.b_conv1)
            
            
            #pooling layer 1
            self.pool1 =  max_pool(self.conv1, 3, 2)
            

            #fully conv layer 1
            self.w_fc1 = weight_variable([size[0]/2 * size[1]/2 * 64, 128],lr_type = 'fc')
            self.b_fc1 = bias_variable([128])
            self.pool1_flat = tf.reshape(self.pool1, [-1, size[0]/2 * size[1]/2 *64])
            self.fc1 = tf.nn.relu(tf.matmul(self.pool1_flat, self.w_fc1) + self.b_fc1)
            
            
            #concat
            self.concat1 = tf.concat([self.fc1,self.from_12],1)


            #fully conv layer 2
            self.w_fc2 = weight_variable([128+16, 1],lr_type = 'fc')
            self.b_fc2 = bias_variable([1])
            self.fc2 = tf.matmul(self.concat1,self.w_fc2) + self.b_fc2
            
        if is_train:
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2,labels =self.targets))
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)   
    
    def get_fc(self,inputs_24,net12_fc):
        return self.concat1.eval(feed_dict = {self.inputs:inputs_24,self.from_12:net12_fc})
    def evaluate(self,inputs_24,targets,net_12_fc):
        predict = tf.to_float(tf.greater(tf.nn.sigmoid(self.fc2),0.5))
        label   =  targets
        eva = tf.cast(tf.equal(predict,label),"float").eval(feed_dict = {self.inputs:inputs_24, self.targets:targets,self.from_12:net_12_fc})
        return eva
        
#48-net
class detect_48Net:
    def __init__(self,size = (48,48,3) ,lr = 0.001, is_train = True):
        
        # data,label
        self.inputs = tf.placeholder("float",[None,size[0],size[1],size[2]])
        self.targets = tf.placeholder("float", [None,1])
        # the concat1 from 24net
        self.from_24 = tf.placeholder("float",[None,16+128])

        with tf.variable_scope("48det_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,size[2],64],"w_conv1")
            self.b_conv1 = bias_variable([64],"b_conv1")
            self.conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, 1) + self.b_conv1)
            
            
            #pooling layer 1
            self.pool1 =  max_pool(self.conv1, 3, 2)
            

            #conv layer 2
            self.w_conv2 = weight_variable([5,5,64,64],"w_conv2")
            self.b_conv2 = bias_variable([64],"b_conv2")
            self.conv2 = tf.nn.relu(conv2d(self.pool1, self.w_conv2, 1) + self.b_conv2)
            
            
            #pooling layer 2
            self.pool2 =  max_pool(self.conv2, 3, 2)
            


            #fully conv layer 1
            self.w_fc1 = weight_variable([size[0]/4 * size[1]/4 * 64, 256],lr_type = 'fc')
            self.b_fc1 = bias_variable([256])
            self.pool2_flat = tf.reshape(self.pool2, [-1, size[0]/4 * size[1]/4 *64])
            self.fc1 = tf.nn.relu(tf.matmul(self.pool2_flat, self.w_fc1) + self.b_fc1)
            
            
            #concat
            
            self.concat1 = tf.concat([self.fc1,self.from_24],1)


            #fully conv layer 2
            self.w_fc2 = weight_variable([256+128+16, 1],lr_type = 'fc')
            self.b_fc2 = bias_variable([1])
            self.fc2 = tf.matmul(self.concat1,self.w_fc2) + self.b_fc2
        if is_train:
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2,labels =self.targets))
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)  
    def evaluate(self,inputs_48,targets,net_24_fc):
        predict = tf.to_float(tf.greater(tf.nn.sigmoid(self.fc2),0.5))
        label   =  targets
        eva = tf.cast(tf.equal(predict,label),"float").eval(feed_dict = {self.inputs:inputs_48, self.targets:targets,self.from_24:net_24_fc})
        return eva



class calib_12Net:
# notice : change the size of 24 net because of shape(12*12) is too small to predict pattern
    def __init__(self,size = (48,48,3) ,lr = 0.001, is_train = True):
        # data,label
        self.inputs = tf.placeholder("float",[None,size[0],size[1],size[2]])
        self.targets = tf.placeholder("float", [None,45])

        #12-net
        with tf.variable_scope("12calib_"):
            #conv layer 1
            self.w_conv1 = weight_variable([3,3,size[2],16],"w1")
            self.b_conv1 = bias_variable([16],"b1")
            self.conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.pool1 =  max_pool(self.conv1, 3, 2)

            #fc layer 1
            self.w_fc1 =  weight_variable([size[0]//2 * size[1]//2 * 16, 128],"w2")
            self.b_fc1 =  bias_variable([128],"b2")
            self.pool1_reshaped = tf.reshape(self.pool1, [-1, size[0]//2 * size[1]//2 * 16])
            self.fc1 = tf.nn.relu(tf.matmul(self.pool1_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([128, 45],"w3")
            self.b_fc2 =  bias_variable([45],"b3")
            self.fc2 = tf.matmul(self.fc1, self.w_fc2) + self.b_fc2
        if is_train:
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2,labels =self.targets))
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)   
    def evaluate(self,inputs_12,targets):
        predict = tf.argmax( self.fc2,1)
        label   =  tf.argmax(targets,1)
        eva = tf.cast(tf.equal(predict,label),"float").eval(feed_dict = {self.inputs:inputs_12, self.targets:targets})
        return eva

class calib_24Net:
    # notice : change the size of 24 net because of shape(24*24) is too small to predict pattern
    def __init__(self,size = (48,48,3) ,lr = 0.001, is_train = True):
        # data,label
        self.inputs = tf.placeholder("float",[None,size[0],size[1],size[2]])
        self.targets = tf.placeholder("float", [None,45])
        

        #24-net
        with tf.variable_scope("24calib_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,size[2],32],"w1")
            self.b_conv1 = bias_variable([32],"b1")
            self.conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.pool1 =  max_pool(self.conv1, 3, 2)

            #fc layer 1
            self.w_fc1 =  weight_variable([size[0]//2 * size[1]//2 * 32, 64],"w2")
            self.b_fc1 =  bias_variable([64],"b2")
            self.pool1_reshaped = tf.reshape(self.pool1, [-1, size[0]//2 * size[1]//2  * 32])
            self.fc1 = tf.nn.relu(tf.matmul(self.pool1_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([64, 45],"w4")
            self.b_fc2 =  bias_variable([45],"b4")
            self.fc2 = tf.matmul(self.fc1, self.w_fc2) + self.b_fc2

        if is_train:
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2,labels =self.targets))
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)    
    def evaluate(self,inputs_24,targets):
        
        predict = tf.argmax(self.fc2,1)
        label   =  tf.argmax(targets,1)
        eva = tf.cast(tf.equal(predict,label),"float").eval(feed_dict = {self.inputs:inputs_24, self.targets:targets})

        return eva

class calib_48Net:

    def __init__(self,size = (48,48,3) ,lr = 0.001, is_train = True):

        # data,label
        self.inputs = tf.placeholder("float",[None,size[0],size[1],size[2]])
        self.targets = tf.placeholder("float", [None,45])
        
        #24-net
        with tf.variable_scope("48calib_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,size[2],64],"w1")
            self.b_conv1 = bias_variable([64],"b1")
            self.conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.pool1 =  max_pool(self.conv1, 3, 2)
            
            #conv layer 2
            self.w_conv2 = weight_variable([5,5,64,64],"w2")
            self.b_conv2 = bias_variable([64],"b2")
            self.conv2 = tf.nn.relu(conv2d(self.pool1, self.w_conv2, 1) + self.b_conv2)

            #fc layer 1
            self.w_fc1 =  weight_variable([size[0]//2 * size[1]//2 * 64, 256],"w3")
            self.b_fc1 =  bias_variable([256],"b3")
            self.conv2_reshaped = tf.reshape(self.conv2, [-1, size[0]//2 * size[1]//2 * 64])
            self.fc1 = tf.nn.relu(tf.matmul(self.conv2_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([256, 45],"w4")
            self.b_fc2 =  bias_variable([45],"b4")
            self.fc2 = tf.matmul(self.fc1, self.w_fc2) + self.b_fc2
        if is_train:
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2,labels =self.targets))
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)    
         
    def evaluate(self,inputs_48,targets):
        predict = tf.argmax(self.fc2,1)
        label   =  tf.argmax(targets,1)
        eva = tf.cast(tf.equal(predict,label),"float").eval(feed_dict = {self.inputs:inputs_48, self.targets:targets})
        return eva
