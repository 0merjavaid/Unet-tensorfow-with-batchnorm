from data1 import DataHandler as DH
import os.path
from scipy.ndimage import rotate
import tensorflow as tf
import numpy as np
import cv2
import scipy.misc as misc



class UNET:
    def __init__(self):
        pass
    
    def model(self,image,keep_prob):
         
        self.conv1_1=self.conv(image,1,	3,64,"conv1_1")
        self.conv1_2=self.conv(self.conv1_1,64,3,64,"conv1_2")
        self.pool1=self.max_pool(self.conv1_2,"pool1")
        
        self.conv2_1=self.conv(self.pool1,64,3,128,"conv2_1")
        self.conv2_2=self.conv(self.conv2_1,128,3,128,"conv2_2")
        self.pool2=self.max_pool(self.conv2_2,"pool2")
        
        self.conv3_1=self.conv(self.pool2,128,3,256,"conv3_1")
        self.conv3_2=self.conv(self.conv3_1,256,3,256,"conv3_2")
        self.pool3=self.max_pool(self.conv3_2,"pool3")
        
        self.conv4_1=self.conv(self.pool3,256,3,512,"conv4_1")
        self.conv4_2=self.conv(self.conv4_1,512,3,512,"conv4_2")
        self.pool4=self.max_pool(self.conv4_2,"pool4")
        
        self.conv5_1=self.conv(self.pool4,512,3,1024,"conv5_1")
        self.conv5_2=self.conv(self.conv5_1,1024,3,1024,"conv5_2") 
        
	self.conv5_2 = tf.nn.dropout(self.conv5_2,keep_prob=keep_prob)
        
        
        
        
        
        
        
        self.up1=self.up_conv(self.conv5_2,"up_conv_1")
        self.up1=self.crop_and_concat(self.conv4_2,self.up1)
        self.conv6_1=self.conv(self.up1,1024,3,512,"conv6_1")
        self.conv6_2=self.conv(self.conv6_1,512,3,512,"conv6_2")
        
        self.up2=self.up_conv(self.conv6_2,"up_conv_2")
        self.up2=self.crop_and_concat(self.conv3_2,self.up2)  
        self.conv7_1=self.conv(self.up2,512,3,256,"conv7_1")
        self.conv7_2=self.conv(self.conv7_1,256,3,256,"conv7_2")
        
        self.up3=self.up_conv(self.conv7_2,"up_conv_3")
        self.up3=self.crop_and_concat(self.conv2_2,self.up3)
        self.conv8_1=self.conv(self.up3,256,3,128,"conv8_1")
        self.conv8_2=self.conv(self.conv8_1,128,3,128,"conv8_2")
        
        self.up4=self.up_conv(self.conv8_2,"up_conv_4")
        self.up4=self.crop_and_concat(self.conv1_2,self.up4)
        self.conv9_1=self.conv(self.up4,128,3,64,"conv9_1")
        self.conv9_2=self.conv(self.conv9_1,64,3,64,"conv9_2")
        self.out=self.conv(self.conv9_2,64,1,2,"out")
        self.pred=tf.argmax(self.out, dimension=3, name="Pred")

        print "Model created"
        
        
        




        
    def max_pool(self,input,name):
        return tf.nn.max_pool(value=input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name=name)
    
    def conv(self,input,num_input_channel,filter_size,num_filter,name,use_relu=True):
        weight_shape=[filter_size,filter_size,num_input_channel,num_filter]
        weights=self.weight(weight_shape,name)
        biases=self.bias(num_filter,name)
        layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding="VALID")
        final = tf.nn.bias_add(layer,biases)
        if(use_relu):
            final = tf.nn.relu(final)
            
        return final
        
    def up_conv(self,input,name):
        stride=[1,2,2,1]
        shape=input.get_shape().as_list()
        weight_shape=[2,2,shape[3]/2 , shape[3]]
        weights=self.weight(weight_shape,name,"up")
        out_shape=[shape[0],shape[1]*2,shape[2]*2,shape[3]/2]
        
        layer=tf.nn.conv2d_transpose(value=input,filter=weights,output_shape=out_shape,strides=stride,padding="VALID")
        
        return layer
    
    def copy_and_crop(self,):
        pass
    def weight(self,shape,name,type="down"):
        if(type=="down"):
	    feature=shape[2]
	else:
	    feature=shape[3]
        stddev=np.sqrt((float(2)/(shape[0]*shape[0]*feature)))
	return tf.Variable(tf.truncated_normal(shape,stddev=stddev),name="weights_"+name)

    def bias(self,shape,name):
	
        return tf.Variable(tf.constant(0.05,shape=[shape]),name="bias_"+name)
    def crop_and_concat(self,x1,x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


    
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
y=tf.placeholder(tf.float32,shape=[1,580,580,2],name="label")
learning=tf.placeholder(tf.float32)
Net.model(x,keep_prob)
saver = tf.train.Saver()



eps = 1e-5
prediction = pixel_wise_softmax_2(Net.out)
intersection = tf.reduce_sum(prediction * y)
union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
Loss = (200 * intersection/ (union))



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
            saver.restore(sess, "ckpt/model2.ckpt")
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
            saver.save(sess,"ckpt/model2.ckpt")
            misc.imsave("file2.png",np.squeeze(im1).astype("uint8")*255)
            print "Saved"



