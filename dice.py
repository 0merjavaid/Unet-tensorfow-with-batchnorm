from data1 import DataHandler as DH
import os.path
from scipy.ndimage import rotate
import tensorflow as tf
import numpy as np
import cv2
import scipy.misc as misc
from model import UNET as UNET




def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)    
    




tf.reset_default_graph()

Net=UNET()
import glob 
keep_prob=tf.placeholder(tf.float32)
x=tf.placeholder(tf.float32,shape=[1,764,764,1],name="image")
y=tf.placeholder(tf.int32,shape=[1,580,580,2],name="label")
learning=tf.placeholder(tf.float32)
Net.model(x,keep_prob)
saver = tf.train.Saver()



gt=tf.reshape(tf.squeeze(y),shape=[-1,2])
result=tf.reshape(Net.out,shape=[-1,2])
weight_matrix=tf.constant([1,2],dtype=tf.int32)
weight_map=tf.cast(tf.reduce_sum(tf.multiply(gt,weight_matrix),axis=1),tf.float32)


#Loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(y,squeeze_dims=[3]), logits=Net.out,name="Loss")))  # Define loss fu$
loss=tf.nn.softmax_cross_entropy_with_logits(labels=gt,logits=result)

Loss=tf.reduce_mean(tf.multiply(loss,weight_map))
#predicter = pixel_wise_softmax_2(Net.out)
#correct_pred = tf.equal(tf.argmax(tf.cast(predicter,tf.int32), 3), y)
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


  






im=lab=None
def get_batch(i):
    global im,lab
    name="train2.png"
    image=misc.imread(name)
    name="test2.png"
    label=misc.imread(name)
    
    image = cv2.resize(image, (572, 572))
    label=cv2.resize(label,(572,572))
    im= image.reshape([572,572,3])
    lab=label[:,:,0]

def train(loss_val,learning):
    optimizer = tf.train.AdamOptimizer(learning)
    grads = optimizer.compute_gradients(loss_val)
    return optimizer.apply_gradients(grads)


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]


 
train_op = train(Loss,learning)
train_set=DH()
test_set=DH()
train_set.read_data("train/","train_label/")
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
    print "something"
    learn=0.000001
    val=10
    wait=22
    avg_loss=[]
    for i in range(0,100000):
        out=train_set.get_next_batch(1,train=True)
        image=out[0][0].reshape(1,764,764,1)
	labels=out[0][1].astype("uint8")
	label=np.zeros((764,764,2)).astype("int32")
	label[:,:,1]=labels
	label[:,:,0]=~labels
	label=(label/255).reshape((1,764,764,2)).astype("uint8")
        if((i+1)%5==0 and val>53):
	    learn/=val
            val=val/5
            
	if((i+1)%100000000==0):
	   learn=0.00001
	   print learn
	

        if(i%20==0):
	    out=test_set.get_next_batch(1,train=False)
	    test_image=out[0][0].reshape([1,764,764,1])
	    test_label=out[0][1].reshape([1,764,764,1])
            train_loss=sess.run(Loss,feed_dict={x:image,y:crop_to_shape(label,[1,580,580,2]),keep_prob:0.65})
	    #test_loss=sess.run(Loss,feed_dict={x:test_image,y:crop_to_shape(test_label,[1,580,580,1]),keep_prob:0.65})
	    #acc=sess.run(accuracy,feed_dict={x:test_image,y:crop_to_shape(test_label,[1,580,580,1]),keep_prob:1.})	    
	    avg_loss.append(train_loss)
	    if(len(avg_loss)>20):
		avg_loss=avg_loss[1:]

            print i, "    Train Loss: ",train_loss,"      Avg Loss", float(sum(avg_loss))/20
	sess.run(train_op,feed_dict={x:image,y:crop_to_shape(label,[1,580,580,2]),keep_prob:0.65,learning:learn})
           
        if((i+1)%200==0):
            im1=sess.run(Net.pred,feed_dict={x:image,keep_prob:1.0})
            saver.save(sess,"ckpt/back2white.ckpt")
            misc.imsave("file3.png",np.squeeze(im1).astype("uint8")*255)
            print "Saved"



