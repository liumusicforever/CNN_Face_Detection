# CNN_Face_Detection
Repository for "A Convolutional Neural Network Cascade for Face Detection", implemented with Python interface.

## About
This repo implemented the [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf) in python/tensorflow , providing the interface to contruct the cascade structure , including function of **detection networks** , **calibration networks** , **image pyramids** and **non maximum supression** .

## Requirement
* Thensorflow : [Tensorflow installation guild](https://www.tensorflow.org/install/)
* Opencv : [OpenCV installation guild](https://pypi.python.org/pypi/opencv-python)

## Useful Operations
* Import Detector
```python
# make sure detection.py is in program folder
from  detection import Detector
```
* Restore the pretrained model

```python
# given paths from both models to Detector , and It will load the model on 
# your memory (or gpu memory).
det_mod_path = 'models/det_net_<epoch num>.ckpt'
cal_mod_path = 'models/cal_net_<epoch num>.ckpt'
detector = Detector(det_mod_path,cal_mod_path)
```

* Processing image pyramids
```python
# bboxes is all bounding boxes of sliding windows , It’s include position
# and probability of face (default is -0.1)
# bboxes = [<xmin> , <ymin> , <xmax> , <ymax> , <probability>]
bboxes = detector.img_pyramids(image)
```
* Non Maximum Suppression
```python
# iou_thresh is the overlapping threshold of iou in non maximum suppression 
# In returning bboxes , function will set the box’s probability = 0.0 which have 
# been filtered.
bboxes = detector.non_max_sup(bboxes,iou_thresh = 0.5)
```

* Predict the bounding boxes on detection/calibration net
```python
# predict function will predict all the bounding boxes which probability is not 
# zero , function will set the box’s probability from prediction and return the 
# final bboxes .
# flags of net :  ‘net12’ , ‘net24’ , ‘net48’ , ‘net12_cal’ , ‘net24_cal’ , ‘net48_cal’ .
# threshold : the threshold of preditction.
bboxes = detector.predict(img,bboxes,net = 'net12',threshold = 0.9)
```

### Results
![image](https://github.com/liumusicforever/CNN_Face_Detection/blob/master/data/results/img_1_result.jpg)


## Implementation Issue
### 12-net and 24-net is too small ?
When I was training models , **finding size of 12-net and 24-net was so hard to convergence** , maybe the size of network is too small to learn pattern , so I change network size of net12 and net24 to 48*48 finally. But still confuse about it !
## Necessary for calibration network ?
The accuracy of the calibration almost only 0.8 , **It result the calibration of bounding box after network may making mistake , and bounding box will be removed in next stage** , so sometimes I have better result without calibration net.


## License

MIT LICENSE

## Reference

Haoxiang Li, Zhe Lin, Xiaohui Shen, Jonathan Brandt, Gang Hua ; A Convolutional Neural Network Cascade for Face Detection ; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 5325-5334