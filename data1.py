import cv2 , os,glob,sys
import numpy as np
import scipy.misc as misc
from scipy.ndimage import rotate
from random import shuffle


i=0
class DataHandler(object):
    
    def __init__(self):
        self.current_batch_index=0
        self.data_length=0
        self.images=[]
    def read_data(self,image_path,label_path):
        
         
        original= os.getcwd()
        dataset_path=os.getcwd()+"/"+image_path
        os.chdir(dataset_path)
	

        for file in glob.glob("*.png"):
            image=cv2.imread(file,0)
            label_name=original+"/"+label_path+file
            label=cv2.imread(label_name)
            self.images.append((image,label))
        os.chdir(original)
        self.data_length=len(self.images)
	
	
        return self.images
    
    def get_next_batch(self,batch_size,train=True):
       
        if((self.current_batch_index+batch_size) > self.data_length):
            batch_size=self.data_length-self.current_batch_index
        
        next_batch=self.images[self.current_batch_index:self.current_batch_index+batch_size]
        self.current_batch_index+=batch_size
       
        if(self.current_batch_index>=self.data_length):
            self.current_batch_index=0
            shuffle(self.images)
         
        if(train):
            next_batch=self.augment(next_batch)
        else:
            next_batch=self.clean(next_batch)
	return next_batch
    def augment(self,batch):
        global i
        temp=[]
        for tuple in batch:
            image=tuple[0]
            label=tuple[1]
   	    label[label>=128]=255
            label[label<128]=0
            label=label[:,:,1]
            angle=np.random.randint(0,360)
            image=rotate(image,angle)
            label=rotate(label,angle)
            if(angle%3==0):
                crop_shape=[image.shape[0]/2,image.shape[1]/2]
                crop_rows=np.random.randint(0,image.shape[0]/2)
                crop_cols=np.random.randint(0,image.shape[1]/2)
                 
                image=image[crop_rows:crop_rows+crop_shape[0],crop_cols:crop_cols+crop_shape[1]]
                label=label[crop_rows:crop_rows+crop_shape[0],crop_cols:crop_cols+crop_shape[1]]
                                
                
                
                
            
            resize_factor=np.random.randint(550,764) 
            image=cv2.resize(image ,(resize_factor,resize_factor))
            label=cv2.resize(label ,(resize_factor,resize_factor) )

            _,label = cv2.threshold(label,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            contrast_level=np.random.uniform(1,4)
            grid_size=np.random.randint(5,20)
            clahe = cv2.createCLAHE(clipLimit=contrast_level, tileGridSize=(grid_size,grid_size))
            image = clahe.apply(image)

            final_image=np.zeros((764,764))
            final_label=np.zeros((764,764))
            
            shape=image.shape
            
            gap=int(float(764-shape[0])/2)
            
            final_image[gap:gap+shape[0],gap:gap+shape[0]]=image
            final_label[gap:gap+shape[0],gap:gap+shape[0]]=label*255
            
            temp.append((final_image,final_label))
        return temp
            
             
    def clean(self,batch):
        temp=[]
        for tuple in batch:
            image=tuple[0]
            label=tuple[1]
            label[label>=128]=255
            label[label<128]=0
            label=label[:,:,1]
            resize_factor=580
            image=cv2.resize(image ,(resize_factor,resize_factor))
            label=cv2.resize(label ,(resize_factor,resize_factor) )

            _,label = cv2.threshold(label,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
            image = clahe.apply(image)
            
            final_image=np.zeros((764,764))
            final_label=np.zeros((764,764))
            
            shape=image.shape
            
            gap=int(float(764-shape[0])/2)
            
            final_image[gap:gap+shape[0],gap:gap+shape[0]]=image
            final_label[gap:gap+shape[0],gap:gap+shape[0]]=label*255
             
            temp.append((final_image,final_label))
        return temp
            
