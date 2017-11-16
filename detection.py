'''
Implementation of "A Convolutional Neural Network Cascade for Face Detection "
Paper : https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
Author : Dennis Liu
Modify : 2017/11/10

Description :   Three important class use to detect img : 
                Classifier : create the detection net 12,24,48 , and provide each net predict function
                Aligner :  create the calibration net 12,24,48 , and provide each net predict function
                Detector : include non max suppression ,image pyramids , interface to use Classifer & Aliner

'''


import cv2
import numpy as np
import tensorflow as tf

import model


class Classifier:
    def __init__(self,model_path,sizes = [12,24,48]):
        self.sizes = sizes
        # load network
        self.net_12 = model.detect_12Net(is_train = False,size = (sizes[0],sizes[0],3))  
        self.net_24 = model.detect_24Net(is_train = False,size = (sizes[1],sizes[1],3))
        self.net_48 = model.detect_48Net(is_train = False)
        # create session
        self.sess = tf.Session()
        self.restore(model_path)
    def restore(self,model_path):
        all_var =   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='12det_')+\
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='24det_')+\
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='48det_')
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(all_var)
        # Restore model from disk.
        saver.restore(self.sess, model_path)
    def net_12_predict(self,data,threshold = 0.5):
        # resize data to fit the size of net work
        input_12 = np.array([cv2.resize(img,(self.sizes[0],self.sizes[0]))for img in data])
        
        # the class of prediction
        max_idx = tf.to_float(tf.argmax( self.net_12.fc2,1))
        # the confidence of predicted class
        max_value = tf.reduce_max(tf.nn.softmax(self.net_12.fc2), axis=1)
        # combine the result
        predict = tf.stack([max_idx,max_value],1)

        # forward 
        result = self.sess.run(predict,feed_dict = {self.net_12.inputs : input_12})  
        return result

    def net_24_predict(self,data,threshold = 0.5):
        # resize data to fit the size of net work
        input_12 = np.array([cv2.resize(img,(self.sizes[0],self.sizes[0]))for img in data])
        input_24 = np.array([cv2.resize(img,(self.sizes[1],self.sizes[1]))for img in data])
        
        # get previous net output
        net_12_fc = self.sess.run(self.net_12.fc1,feed_dict = {self.net_12.inputs :input_12})

        max_idx = tf.to_float(tf.argmax( self.net_24.fc2,1))
        max_value = tf.reduce_max(tf.nn.softmax(self.net_24.fc2), axis=1)
        predict = tf.stack([max_idx,max_value],1)
        
        result = self.sess.run(predict,feed_dict = {self.net_24.inputs : input_24, self.net_24.from_12 : net_12_fc})
        return result
    def net_48_predict(self,data,threshold = 0.5):
        # resize data to fit the size of net work
        input_12 = np.array([cv2.resize(img,(self.sizes[0],self.sizes[0]))for img in data])
        input_24 = np.array([cv2.resize(img,(self.sizes[1],self.sizes[1]))for img in data])
        input_48 = np.array([cv2.resize(img,(self.sizes[2],self.sizes[2]))for img in data])
        
        # get previous net output
        net_12_fc = self.sess.run(self.net_12.fc1,feed_dict = {self.net_12.inputs :input_12})
        net_24_fc = self.sess.run(self.net_24.concat1,feed_dict = {self.net_24.inputs : input_24, self.net_24.from_12 : net_12_fc})

        # the class of prediction
        max_idx = tf.to_float(tf.argmax( self.net_48.fc2,1))
        # the confidence of predicted class
        max_value = tf.reduce_max(tf.nn.softmax(self.net_48.fc2), axis=1)
        # combine the result
        predict = tf.stack([max_idx,max_value],1)

        result = self.sess.run(predict,feed_dict = {self.net_48.inputs : input_48, self.net_48.from_24 : net_24_fc})
        return result

class Aligner:
    def __init__(self,model_path,sizes = [12,24,48]):
        self.sizes = sizes
        # load network
        self.net_12 = model.calib_12Net(is_train = False,size = (sizes[0],sizes[0],3))
        self.net_24 = model.calib_24Net(is_train = False,size = (sizes[1],sizes[1],3))
        self.net_48 = model.calib_48Net(is_train = False,size = (sizes[2],sizes[2],3))
        # create session
        self.sess = tf.Session()
        self.restore(model_path)
    def restore(self,model_path):
        all_var =   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='12calib_')+\
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='24calib_')+\
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='48calib_')
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(all_var)
        # Restore model from disk.
        saver.restore(self.sess, model_path)
        
    def net_12_predict(self,data):
        # resize data to fit the size of net work
        input_12 = np.array([cv2.resize(img,(self.sizes[0],self.sizes[0]))for img in data])
        
        predict = tf.argmax( self.net_12.fc2,1)
        result = self.sess.run(predict,feed_dict = {self.net_12.inputs : input_12})
        return result
    def net_24_predict(self,data):
        # resize data to fit the size of net work
        input_24 = np.array([cv2.resize(img,(self.sizes[1],self.sizes[1]))for img in data])
        

        predict = tf.argmax( self.net_24.fc2,1)
        result = self.sess.run(predict,feed_dict = {self.net_24.inputs : input_24})
        return result
    def net_48_predict(self,data):
        # resize data to fit the size of net work
        input_48 = np.array([cv2.resize(img,(self.sizes[2],self.sizes[2]))for img in data])
        
        predict = tf.argmax( self.net_48.fc2,1)
        result = self.sess.run(predict,feed_dict = {self.net_48.inputs : input_48})
        return result

class Detector:
    def __init__(self,det_path,cal_path,pyramid_t = 3,win_size = (48,48),win_stride =  10):
        # configuration of detection windows
        self.pyramid_t = pyramid_t
        self.win_size = win_size
        self.win_stride =  win_stride
        # config of network
        self.batch = 1000
        
        # load the models
        self.classifier = Classifier(det_path)
        self.aligner = Aligner(cal_path)

        self.result = []
    def detect(self,img):
        '''
            step 1. do pyramid and detect in sliding windows
            step 2. consturct the cascade structure
        '''
        # net 12 classifier 
        
        # net 24 classifier 
        
        # net 48 classifier
        
        return 
    def predict(self,img,bboxes,net = None,threshold = 0.9): 
        batch = self.batch
        win_buff = []
        idx_buff = []
        h,w = img.shape[:2]
        
        # mapping the bboxes to generate batch
        for idx,bbox in enumerate(bboxes):
            xmin,ymin,xmax,ymax,prop = bbox[:]
            if prop == 0.0 : continue
            win = img[int(ymin*h):int(ymax*h),int(xmin*w):int(xmax*w)]
            if win is None or win.shape[0] < 1 or win.shape[1] < 1 : continue
            win = cv2.cvtColor(cv2.resize(win,(48,48)),cv2.COLOR_BGR2RGB)
            win_buff.append(win)
            idx_buff.append(idx)

            if len(win_buff)>=batch:
                bboxes = self.net_forward(win_buff,idx_buff,bboxes,net,threshold)
                win_buff = []
                idx_buff = []
        if len(win_buff) > 0:
            bboxes = self.net_forward(win_buff,idx_buff,bboxes,net,threshold)
        return bboxes
    def net_forward(self,win_buff,idx_buff,bboxes,net , threshold):
        # forward the detection net
        if not 'cal' in net :
            if net == 'net12':
                res = self.classifier.net_12_predict(win_buff,threshold)
            elif net == 'net24':
                res = self.classifier.net_24_predict(win_buff,threshold)
            elif net == 'net48':
                res = self.classifier.net_48_predict(win_buff,threshold)
            else:
                return None
            for i,idx in enumerate(idx_buff):
                is_face,prop = res[i]
                if is_face == 1.0:
                    bboxes[idx][4] = prop
                else:
                    bboxes[idx][4] = 0.0
            return bboxes
        else:
            # forward the calibration net
            if net == 'net12_cal':
                res = self.aligner.net_12_predict(win_buff)
            elif net == 'net24_cal':
                res = self.aligner.net_24_predict(win_buff)
            elif net == 'net48_cal':
                res = self.aligner.net_48_predict(win_buff)
            else:
                return None
            cali_scale = [1.20, 1.09, 1.0, 0.9, 0.82]
            cali_off_x = [0.17, 0., -0.17]
            cali_off_y = [0.17, 0., -0.17]

            for i,idx in enumerate(idx_buff):
                clss = res[i]
                h,w = win_buff[i].shape[:2]
                xmin,ymin,xmax,ymax = bboxes[idx][:4]
                xmin,ymin,xmax,ymax = xmin*w,ymin*h,xmax*w,ymax*h
                s = cali_scale[int(clss/(len(cali_off_x)*len(cali_off_y)))]
                x_off = cali_off_x[int(s/len(cali_off_y))]
                y_off = cali_off_y[int(s%len(cali_off_y))]
                new_xmin = xmin - x_off*(xmax-xmin)/s
                new_ymin = ymin - y_off*(ymax-ymin)/s
                new_xmax = new_xmin+(xmax-xmin)/s
                new_ymax = new_ymin+(ymax-ymin)/s
                bboxes[idx][:4] = new_xmin/w,new_ymin/h,new_xmax/w,new_ymax/h
            return bboxes
            
    def non_max_sup(self,bboxes,iou_thresh = 0.5):
        def overlap(box1,box2):
            
            # determine the coordinates of the intersection rectangle
            in_xmin = max([box1[0],box2[0]])
            in_ymin = max([box1[1],box2[1]])
            in_xmax = min([box1[2],box2[2]])
            in_ymax = min([box1[3],box2[3]])

            if in_xmax < in_xmin or in_ymax < in_ymin:
                return 0.0 , 0.0 , 0.0

            # compute the intersection area
            intersection_area = (in_xmax - in_xmin) * (in_ymax - in_ymin)

            
            # compute the area of both bboxes
            box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
            box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            
            box1_iou = intersection_area / float(box1_area)
            box2_iou = intersection_area / float(box2_area)
            
            return iou,box1_iou,box2_iou

        # bboxes = [<xmin>,<ymin>,<xmax>,<ymax>,<prop>]
        for i,bbox1 in enumerate(bboxes) :
            if bbox1[4] < 0.0001 : continue
            for j,bbox2 in enumerate(bboxes):
                if bbox2[4] < 0.0001 : continue
                if i==j : continue
                iou,box1_iou,box2_iou = overlap(bbox1,bbox2)
                bbox1_prop = bbox1[4]
                bbox2_prop = bbox2[4]
                # # inner box threshold
                # if box1_iou > 0.9:
                #     bbox1[4] = 0.0
                # elif box2_iou > 0.9:
                #     bbox2[4] = 0.0
                if iou >= iou_thresh:
                    if bbox1_prop <= bbox2_prop:
                        bbox1[4] = 0.0
                    elif bbox1_prop > bbox2_prop:
                        bbox2[4] = 0.0
        return bboxes

    def img_pyramids(self,img):
        # init the return list
        # bbox = [<xmin>,<xmax>,<ymin>,<ymax>,<prop>]
        bboxes = []
        # slide a window across the image
        def sliding_window(image, stepSize, windowSize):
            for y in range(0, image.shape[0], stepSize):
                for x in range(0, image.shape[1], stepSize):
                    # yield the current window
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
        
        # set sliding windows config
        win_stride = self.win_stride
        win_size = self.win_size

        # generate Gaussian pyramid for img
        imgPyramids = [img.copy()]
        for i in range(1, self.pyramid_t):
            imgPyramids.append(cv2.pyrDown(imgPyramids[i - 1]))
        # sliding all image from pyramids
        for i in range(self.pyramid_t):
            p_img = imgPyramids[i]
            p_h,p_w = p_img.shape[:2]
            for (x, y, window) in sliding_window(p_img, stepSize=win_stride, windowSize=win_size):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != win_size[0] or window.shape[1] != win_size[1]:
                    continue
                
                x,y = float(x),float(y)
                bbox = [x/p_w,y/p_h,(x+win_size[0])/p_w,(y+win_size[1])/p_h,-0.1]
                bboxes.append(bbox)
        return bboxes
    

        

def test_detect():
    det_mod_path = 'models/48_net_223.ckpt'
    cal_mod_path = 'models/48_cal_net_100.ckpt'
    detector = Detector(det_mod_path,cal_mod_path)

    img_path = '/home/share/data/FDDB/2002/07/25/big/img_362.jpg'
    img_path = '/home/share/data/FDDB/2002/07/25/big/img_1026.jpg'
    img = cv2.imread(img_path)
    h , w = img.shape[:2]
    
    bboxes = detector.detect(img)
    import matplotlib.pyplot as plt
    
    for b in bboxes:
        xmin,ymin,xmax,ymax,prop = b[:]
        if prop > 0.5:
            cv2.rectangle(img, (int(xmin*w), int(ymin*h)), (int(xmax*w), int(ymax*h)), (255, 0, 0), 2)
    plt.imshow(img)
    plt.show()

def test_predict():
    def read_img(img_path,size = (48,48,3)):
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=size[2])
        resized_image = tf.image.resize_images(img_decoded, [size[0], size[1]])
        return resized_image
    
    det_mod_path = 'models/48_cal_net_100.ckpt'
    detector = Aligner(det_mod_path)

    # list of nagative samples paths
    neg_samples = []
    # list of positive samples paths
    pos_samples = []

    
    img_paths = neg_samples + pos_samples
    

    batch = tf.stack([read_img(p,(48,48,3)) for p in img_paths],0)
    data1 = detector.sess.run(batch)

    data = np.array([cv2.cvtColor(cv2.resize(cv2.imread(p),(48,48), interpolation = cv2.INTER_AREA  ),cv2.COLOR_BGR2RGB) for p in img_paths]).astype(np.float32)

    print (detector.net_12_predict(data))
    print (detector.net_12_predict(data))

if __name__ == "__main__":
    # test_predict()
    test_detect()
