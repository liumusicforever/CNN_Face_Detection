import numpy as np
import tensorflow as tf

from tensorflow.contrib.data import Dataset, Iterator


class DataSet:
    def __init__(self, data_path_list , train_rate = 0.9):
        self.data_path_list = data_path_list
        self.train_rate = train_rate
    def get_dataset(self,batch,size = (48,48,3)):
        self.size = size
        dataset = self.data_path_list
        
        from random import shuffle
        shuffle(dataset)

        
        train_set = dataset[0:int(len(dataset)*self.train_rate)]
        val_set = dataset[int(len(dataset)*self.train_rate):]

        # pading last batch
        if len(train_set) % batch != 0 :
            for i in range(batch - (len(train_set) % batch)):
                train_set.append(train_set[0])
        if len(val_set) % batch != 0 :
            for i in range(batch - (len(val_set) % batch)):
                val_set.append(val_set[0])


        train_imgs = tf.constant( [data[0] for data in train_set])
        train_labels = tf.constant([data[1] for data in train_set])

        val_imgs = tf.constant([data[0] for data in val_set])
        val_labels = tf.constant([data[1] for data in val_set])

        # create TensorFlow Dataset objects
        tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
        val_data = Dataset.from_tensor_slices((val_imgs, val_labels))

        tr_data = tr_data.map(self.data_loader)
        val_data = val_data.map(self.data_loader)
        
        return tr_data,val_data

    def get_iterator(self,batch = 3,size = (12,12,3)):
        tr_data , val_data = self.get_dataset(batch,size)

        tr_data = tr_data.batch(batch)
        val_data = val_data.batch(batch)

        # create TensorFlow Iterator object
        iterator = Iterator.from_structure(tr_data.output_types,
                                        tr_data.output_shapes)

        # create two initialization ops to switch between the datasets
        training_init_op = iterator.make_initializer(tr_data)
        validation_init_op = iterator.make_initializer(val_data)

        next_element = iterator.get_next()        
        return iterator , training_init_op , validation_init_op , next_element

    def data_loader(self , img_path, label):
        # label format : [<cls-id>,<pattern-id>]
        

        # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=self.size[2])
        resized_image = tf.image.resize_images(img_decoded, [self.size[0], self.size[1]])
        
        
        classes_num = 2
        clss = tf.one_hot(label[0], classes_num)

        # convert the label to one-hot encoding
        pattern_classes = 45
        pattern = tf.one_hot(label[1], pattern_classes)
        
        return resized_image, clss , pattern


def test_dataset():
    
    dataset = DataSet([['data/img_1.jpg',[0,1]],
                       ['data/img_1.jpg',[1,5]],
                       ['data/img_1.jpg',[0,10]],
                       ])
    _ , train_op , val_op , next_ele = dataset.get_iterator(batch = 1)
    sess = tf.InteractiveSession()
    sess.run(train_op)
    while True:
        try:
            inputs , targets, patterns = sess.run(next_ele)
            # print 'inputs',inputs
            
            print 'targets',targets
            print 'patterns',patterns
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
        


if __name__ == "__main__":
    test_dataset()