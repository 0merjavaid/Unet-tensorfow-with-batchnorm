import os,cv2,time
from model import UNET 
import tensorflow as tf
import numpy as np
from data1 import DataHandler as DH


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]




def accuracy(pred,gt):
    pred=pred.reshape([-1])
    gt=gt.reshape([-1])
    sum=np.sum(pred==gt)
    acc=float(sum)/pred.shape[0]
    return acc

tf.reset_default_graph()

Net=UNET()
import glob 
keep_prob=tf.placeholder(tf.float32)
x=tf.placeholder(tf.float32,shape=[1,764,764,1],name="image")
Net.model(x,keep_prob)
saver = tf.train.Saver()

test_set=DH()
test_set.read_data("test/","test_label/")

init=tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.24))) as sess:
    sess.run(init)
    # you need to initialize all variables
    if(os.path.exists("ckpt/")):
        try:
            saver.restore(sess, "ckpt/back2white.ckpt")
            print "Restored"
        except:
            print  "initialized Model"

    total_time=acc=0
    for i in range(test_set.data_length):
	out=test_set.get_next_batch(1,train=False)
	image=out[0][0].reshape(1,764,764,1)
	labels=np.squeeze(crop_to_shape(out[0][1].reshape(1,764,764,1).astype("uint8"),[1,580,580,1]))


	output=(np.squeeze(sess.run(Net.pred,feed_dict={x:image,keep_prob:1.0}))*255).astype("uint8")
	kernel=np.ones((3,3)).astype("uint8")
	print output.shape
	t=time.time()
	output = cv2.medianBlur(cv2.medianBlur(cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel),3),3)
	now=time.time()-t
	curr=accuracy(output,labels)


	out_name="output/"+str(i)+"pred.png"
	out_label="output/"+str(i)+"label.png"
	
	cv2.imwrite(out_label,labels)
	cv2.imwrite(out_name,output)
        print "Current accuracy:   ",curr,"   Time taken:   ",now
	acc+=curr
	total_time+=now

    print "FINAL ACCURACY : ", float(acc)/test_set.data_length, "FINAL TIME : ",total_time/test_set.data_length

