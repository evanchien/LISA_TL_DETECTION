
# coding: utf-8

# In[98]:


import os, sys
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from random import shuffle


# In[99]:


sys.path.append('/home/evanchien/TF_model/models/research/')
from object_detection.utils import dataset_util


# In[100]:


flags = tf.app.flags
flags.DEFINE_string('train_path', '', 'Path to output training TFRecord')
flags.DEFINE_string('eval_path', '', 'Path to output evaluation TFRecord')
FLAGS = flags.FLAGS


# In[101]:


TRAIN_PATH ='/home/evanchien/lisa_tl/dayTrain/'
DATA_PATH = '/home/evanchien/lisa_tl/'
TRAIN_RATIO = 0.7
subfolders = os.listdir(TRAIN_PATH)


# In[102]:


annotation_dict = {"go":1, "warning":2, "stop":3, "goLeft":4, "warningLeft":5, "stopLeft":6, "goForward":7}



class single_record:
    '''
    A class for the tf_example protos.
    The image data is not loaded and the values are not featurized
    '''

    def __init__(self):
        self.xmins = []
        self.xmaxs = []
        self.ymins = []
        self.ymaxs = []
        self.classes_text = []
        self.classes = []
        # self.encoded_image_data = None
        self.height = 0
        self.width = 0
        self.filename = None


# In[104]:


def img_to_list(manifest, df):
    '''
    Read in - the pandas DataFrame of annotations and file information (Made global here. Can be revised as a real read-in param)
    Return - a list of tf_example prototypes.
    
    '''

    record_list =[]
    item_cnt, _ = df.shape
    current_file = df['Filename'][0]
    record_list.append(single_record())
    for i in range(item_cnt):
        file = df['Filename'][i]
        if file != current_file:
            record_list.append(single_record())
            current_file = file



        filename = os.path.join(manifest, 'frames/',current_file.split('/')[-1])
#         print(filename)
        # with tf.gfile.GFile(filename, 'rb') as fid:
        #     record_list[-1].encoded_image_data = fid.read() 
#         record_list[-1].height, record_list[-1].width = cv2.imread(filename).shape[:2]
        record_list[-1].height = 960
        record_list[-1].width = 1280
        record_list[-1].filename = filename

        # Since bbox are with float type, do transform the int to float
        record_list[-1].xmins.append(df['Upper left corner X'][i]*1.0/record_list[-1].width)
        record_list[-1].xmaxs.append(df['Lower right corner X'][i]*1.0/record_list[-1].width)
        record_list[-1].ymins.append(df['Upper left corner Y'][i]*1.0/record_list[-1].height)
        record_list[-1].ymaxs.append(df['Lower right corner Y'][i]*1.0/record_list[-1].height)
        record_list[-1].classes_text.append(df['Annotation tag'][i].encode())
        record_list[-1].classes.append(annotation_dict[df['Annotation tag'][i]])
    return record_list









def record_generator(records, writer):
    '''
    Create tf records from tf_example prototypes.
    
    Param :
        records - a list of tf_example protos (class single_record)
        writer - the corresponding tf.writer mapped with the flags
    
    Return :
        None but creating two tfrecords file
    '''
    image_format ='png'.encode()

    for record in records:
    # record = records[0]
        with tf.gfile.GFile(record.filename, 'rb') as fid:
            encoded_image_data = fid.read()
        # print("encode ok")

        filename = record.filename.encode()
        # print("filename ok", filename)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(record.height),
            'image/width': dataset_util.int64_feature(record.width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(record.xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(record.xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(record.ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(record.ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(record.classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(record.classes),
            }))
    # print("tf_example ok")
        writer.write(tf_example.SerializeToString())
    writer.close()


# In[ ]:


def main(_):
    
    
    
    # Label_map generation 
    # Exporting the label_map.pbtxt with the dictionary previous generated.
    # Uncomment if needed

    print("Generating label_map")
    with open ("/home/evanchien/lisa_tl/label_map.pbtxt", 'w') as file:
        for key in annotation_dict.keys():
            file.write("item {\n  id: %i\n" %(annotation_dict[key]))
            file.write("  name: '%s'\n}\n\n" %key)
    print("Label map generated")

    sample_list = []
    for subfolder in subfolders:
        manifest = TRAIN_PATH + subfolder
        df = pd.DataFrame(pd.read_csv(manifest+'/frameAnnotationsBOX.csv',sep=',|;', header=0, engine='python'))
        sample_list+=img_to_list(manifest, df)
    # print("sample list size = ", len(sample_list))
    train_size = int(len(sample_list)*TRAIN_RATIO)
    shuffle(sample_list)
    training_list = sample_list[:train_size]
    eval_list = sample_list[train_size:]
    print("Training/Evaluation records generating...")
    

    train_writer = tf.python_io.TFRecordWriter(FLAGS.train_path)
    record_generator(training_list, train_writer)

    # DATA_PATH = "/home/evanchien/lisa/training/"
    # df = pd.DataFrame(pd.read_csv(DATA_PATH+'/allTrainingAnnotations.csv',sep=',|;', header=0, engine='python'))
    print("Training record created")
    eval_writer = tf.python_io.TFRecordWriter(FLAGS.eval_path)
    record_generator(eval_list, eval_writer)
    print("Evaluation record created")
    print("All done!")

    

if __name__ == '__main__':
  tf.app.run()

