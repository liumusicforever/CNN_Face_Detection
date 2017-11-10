'''
This program use to parsing fddb dataset and generate a trining set of 12,24,48 net

author dennisliu
modify 2017/11/10

please modify the out_path and function name of gen_pos_sample or gen_neg_sample.
'''
import os
import uuid
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


fddb_path = '/home/share/data/FDDB/'
label_files = [fddb_path+'FDDB-folds/' + txt for txt in \
['FDDB-fold-01-ellipseList.txt',
'FDDB-fold-02-ellipseList.txt',
'FDDB-fold-03-ellipseList.txt',
'FDDB-fold-04-ellipseList.txt',
'FDDB-fold-05-ellipseList.txt',
'FDDB-fold-06-ellipseList.txt',
'FDDB-fold-07-ellipseList.txt',
'FDDB-fold-08-ellipseList.txt',
'FDDB-fold-09-ellipseList.txt',
'FDDB-fold-10-ellipseList.txt']]

def parse_data_info(only_positive = False,limit_num = None,pos_neg_ratio = 0.5):
    data_info = []
    pos_num = None
    neg_num = None
    import os
    pos_folders = '/home/share/data/FDDB/positive_sample'
    neg_folders = '/home/share/data/FDDB/negative_sample'

    if limit_num:
        pos_num = int(limit_num * pos_neg_ratio)
        neg_num = int(limit_num * (1-pos_neg_ratio))
        poses = os.listdir(pos_folders)[:pos_num]
        negs  = os.listdir(neg_folders)[:neg_num]
    else:
        poses = os.listdir(pos_folders)
        negs = os.listdir(neg_folders)

    for img in poses:
        img_path = os.path.join(pos_folders,img)
        labels = img.replace('.jpg','').split('_')
        clss = int(labels[1])
        pattern = int(labels[2])
        data_info.append([img_path,[clss,pattern]])
    if not only_positive:
        for img in negs:
            img_path = os.path.join(neg_folders,img)
            labels = img.replace('.jpg','').split('_')
            clss = int(labels[1])
            pattern = int(labels[2])
            data_info.append([img_path,[clss,pattern]])
    
    return data_info

def fddb_loader(fddb_path):
    images = []

    for txt in label_files:
        with open(txt) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        idx = 0
        faces = 0
        # convert txt to list
        while idx < len(content):
            if faces == 0:
                filename = fddb_path + content[idx] + '.jpg'
                faces = int(content[idx+1])
                idx += 2
            else:
                bboxes = []
                for i in range(faces):
                    bboxes.append(content[idx+i].split())
                idx += faces
                if os.path.exists(filename) :
                    images.append([filename,faces,bboxes])
                faces = 0
    return images
            
def bbox_convert(images):
    '''
    description : 
        convert Elliptical regions to Rectangular regions
    input : 
        imgaes : [[<filepath>,<faces>,<bboxes>]]
    return : 
        result : [<filepath>,<bboxes>]
    bbox format : 
        bboxes: [<xmin>,<ymin>,<xmax>,<ymax>]
    '''

    result = []
    for i,img in enumerate(images):
        image = cv2.imread(img[0])
        
        # remove when image not avalible
        if image is None: continue

        H,W = image.shape[:2]
        bboxes = []
        for bbox in img[2]:
            h = float(bbox[0])
            x = float(bbox[3])
            w = float(bbox[1])
            y = float(bbox[4])
            xmin = (x-w)/W
            ymin = (y-h)/H
            xmax = (x+w)/W
            ymax = (y+h)/H
            bboxes.append([xmin,ymin,xmax,ymax])
        result.append([img[0],bboxes])
    return result


def show(image,bboxes = None):
    fig,ax = plt.subplots(1)

    img = cv2.imread(image)
    H,W = img.shape[:2]
    if bboxes:
        for bbox in bboxes:
            xmin,ymin,xmax,ymax = bbox[:]
            rect = patches.Rectangle((xmin*W,ymin*H),(xmax-xmin)*W,(ymax-ymin)*H,linewidth=1,fill=False)
            ax.add_patch(rect)
    ax.imshow(img)
    plt.show()

def gen_pos_sample(images , out_path):
    cali_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
    cali_off_x = [-0.17, 0., 0.17]
    cali_off_y = [-0.17, 0., 0.17]

    for image in images:
        im_path = image[0]
        bboxes = image[1]
        img = cv2.imread(im_path)
        H,W = img.shape[:2]
        for bbox in bboxes:
            facename = str(uuid.uuid4())
            for si,s in enumerate(cali_scale):
                for xi,x_off in enumerate(cali_off_x):
                    for yi,y_off in enumerate(cali_off_y):
                        xmin , ymin , xmax , ymax = bbox[:]
                        new_xmin = xmin - x_off*(xmax-xmin)/s
                        new_ymin = ymin - y_off*(ymax-ymin)/s
                        new_xmax = new_xmin+(xmax-xmin)/s
                        new_ymax = new_ymin+(ymax-ymin)/s
                        # crop  
                        face = img[int(new_ymin*H):int(new_ymax*H),int(new_xmin*W):int(new_xmax*W)]
                        
                        if all(i > 10 for i in face.shape[:2]) : 
                            # annot = '{},{},{}'.format(si,xi,yi)
                            clss = xi * len(cali_off_y) + si * len(cali_off_y) * len(cali_off_x) + yi
                            imgname = facename + '_1_' + str(clss) + '.jpg'
                            cv2.imwrite(out_path+'/'+imgname,face)
                            
                            

def gen_neg_sample(images , out_path):
    def check_in_bbox(poses , bboxes):
        '''
        input :
            poses : [<lefttop>,<butt_down>]
            bboxes : [<bbox1>,<bbox2>,...]
        return :
            in_range : 
                True : position of box is in the bboxes of faces
        '''
        in_range = False
        for bbox in bboxes:
            for pos in poses :
                if pos[0] > bbox[0] and pos[0] < bbox[2] and pos[1] > bbox[1] and pos[1] < bbox[3] :
                    in_range = True
                else:
                    pass
        return in_range
    # the background sampleing times
    sample_times = 100
    cali_scale = [0.5, 0.75, 1.0, 1.25, 1.50]

    for image in images:
        im_path = image[0]
        bboxes = image[1]
        img = cv2.imread(im_path)
        H,W = img.shape[:2]
        

        
        for i in range(sample_times):
            # random position
            pos_xmin =  random.uniform(0, 1)
            pos_ymin =  random.uniform(0, 1)

            # set region with position and mean of x abd y
            mean_x = sum([bbox[2]-bbox[0] for bbox in bboxes])/len(bboxes)
            mean_y = sum([bbox[3]-bbox[1] for bbox in bboxes])/len(bboxes)
            for s in cali_scale:
                facename = str(uuid.uuid4())
                pos_xmax = pos_xmin + mean_x/s
                pos_ymax = pos_ymin + mean_y/s
                
                if pos_xmax > 1 or pos_ymax > 1:
                    continue

                poses = [[pos_xmin,pos_ymin],[pos_xmax,pos_ymax],[pos_xmin,pos_ymax],[pos_xmax,pos_ymin],[(pos_xmin+pos_xmax)/2,(pos_ymin+pos_ymax)/2]]
                # check if not region in range of bboxes
                if not check_in_bbox(poses,bboxes):
                    # generate negative sample
                    face = img[int(pos_ymin*H):int(pos_ymax*H),int(pos_xmin*W):int(pos_xmax*W)]
                    imgname = facename + '_0_99.jpg'
                    cv2.imwrite(out_path+'/'+imgname,face)
                    # plt.imshow(face)
                    # plt.show()
                else:
                    continue
        
        

        
            
def main():
    out_path = fddb_path + 'positive_sample/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load FDDB annotation file
    images = fddb_loader(fddb_path)
    print ("total processing image : {}".format(len(images)))
    # convert Elliptical regions to Rectangular regions
    images = bbox_convert(images)
    
    # positive sample generator
    gen_pos_sample(images , out_path)

        
    #image = images[0][0]
    #bboxes = images[0][1]
    # show images
    # show(image,bboxes)

if __name__ == "__main__":
    main()