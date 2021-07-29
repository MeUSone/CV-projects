import cv2
import numpy as np
from matplotlib import pyplot as plt
        
class Grabcut:
    
    def __init__(self,rect):
        self.rect = rect
    
    def render(self,image1,image2):
        
        img1 = cv2.imread(image1)
        
        mask = np.zeros(img1.shape[:2], np.uint8)
        
        cv2.grabCut(img1,mask,self.rect,np.zeros((1, 65), np.float64),np.zeros((1, 65), np.float64),5,cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0), 0, 1)
        
        img1=img1*mask2[:,:,np.newaxis]
        
        img2 = cv2.imread(image2)
        
        img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
 
        for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                for z in range(img1.shape[2]):
                    if (img1[x,y,z]==0):
                        img1[x,y,z]=img2[x,y,z]        
        
        return img1

rect=(84,0,84+428,428)
canvas = Grabcut(rect)
res = canvas.render("trump2.png","biden.jpg")
cv2.imwrite("final.png",res)
        
        