
from scipy.ndimage import rotate
import tensorflow as tf
import numpy as np
import cv2
import scipy.misc as misc

class UNET:
    def __init__(self):
        pass
    
    def model(self,image,keep_prob):
         
        self.conv1_1=self.conv(image,3,3,64,"conv1_1")
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
        weights=self.weight(weight_shape,name)
        out_shape=[shape[0],shape[1]*2,shape[2]*2,shape[3]/2]
        
        layer=tf.nn.conv2d_transpose(value=input,filter=weights,output_shape=out_shape,strides=stride,padding="VALID")
        
        return layer
    
    def copy_and_crop(self,):
        pass
    def weight(self,shape,name):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.05),name="weights_"+name)

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



Net=UNET()
import glob 
keep_prob=tf.placeholder(tf.float32)
x=tf.placeholder(tf.float32,shape=[1,572,572,3],name="image")
y=tf.placeholder(tf.int32,shape=[1,388,388,1],name="label")
learning=tf.placeholder(tf.float32)
Net.model(x,keep_prob)

Loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(y,squeeze_dims=[3]), logits=Net.out,name="Loss")))  # Define loss fu$














im=lab=None
def get_batch(i):
    global im,lab
    name="train1.png"
    image=misc.imread(name)
    name="test1.png"
    label=misc.imread(name)
    
    image = cv2.resize(image, (572, 572))
    label=cv2.resize(label,(572,572))
    im= image.reshape([572,572,3])
    lab=label[:,:,0]

def train(loss_val,learning):
    optimizer = tf.train.AdamOptimizer(learning)
    grads = optimizer.compute_gradients(loss_val)
    return optimizer.apply_gradients(grads)

train_op = train(Loss,learning)


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]



with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    get_batch(1)
    print "something"
    learn=0.0001

    val=10

    for i in range(0,100000):
        image=rotate(im, (180*i)%360, reshape=False).reshape([1,572,572,3])
        label=rotate(lab,(180*i)%360,reshape=False).reshape([1,572,572,1])



        sess.run(train_op,feed_dict={x:image,y:crop_to_shape(label,[1,388,388,2]),keep_prob:0.5,learning:learn})
        if((i+1)%5==0 and val>3):
	    
            learn/=val
	    val=val/5
	    print learn
            
        if(i%1==0):
            print i, "    ",sess.run(Loss,feed_dict={x:image,y:crop_to_shape(label,[1,388,388,2]),keep_prob:0.5})
            
        if(i%400==0):
            im1=sess.run(Net.pred,feed_dict={x:image,y:crop_to_shape(label,[1,388,388,1]),keep_prob:0.5})

            misc.imsave("file.png",im1.reshape([388,388]).astype("uint8")*255)
            print "Saved"

