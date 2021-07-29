import cv2
import numpy as np
import os
import imageio

# This code is major used for landscape photos.

class Cartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.
    """
    def __init__(self):
        pass

    def render(self, img_rgb):
        # read image 
        numDownSamples = 3       # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        # show original image using imshow
        #cv2.imshow("original", img_color)
        #cv2.waitKey(100)
        
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        # show resized image using imshow
        #cv2.imshow("downsampled", img_color)
        #cv2.waitKey(100)
        
        # enhance contrast
        img_color = self.enhanceContrast(img_color)
        
        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        
        # show filtered image using imshow
        #cv2.imshow("filtered", img_color)
        #cv2.waitKey(100)
        # upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)
            
        cv2.imwrite("/Users/user/Downloads/manga_card/landscape_color.png",img_color)
        
        # show upsampled image using imshow
        #cv2.imshow("upsampled", img_color)
        #cv2.waitKey(100)
        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        # show the grey image using imshow
        #cv2.imshow("grey", img_blur)
        #cv2.waitKey(100)

        
        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        # show the edge image using imshow
        #cv2.imshow("edge", img_edge)
        #cv2.waitKey(100)

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        (x,y,z) = img_color.shape
        img_edge = cv2.resize(img_edge,(y,x))
        # convert back to color
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("landscape_edge.png",img_edge)
        # show the step 5 image using imshow
        #cv2.imshow("final", img_edge)
        
        #cv2.waitKey(100)
        
        blackimg = cv2.imread("landscape_edge.png")
        # Level 1 filter
        dst = cv2.fastNlMeansDenoisingColored(blackimg, None, 100, 10, 7, 25)
        # Level 2 filter
        dst = cv2.fastNlMeansDenoisingColored(dst, None, 20, 10, 7, 25)        
        #img_edge = cv2.resize(img_edge,(i for i in img_color.shape[:2]))
        #print img_edge.shape, img_color.shape
        dst = self.cleanup(dst)
        return cv2.bitwise_and(img_color, dst)
    
    def enhanceContrast(self, img_rgb):
        alpha = 2.0
        beta = 0
        new_image = np.zeros(img_rgb.shape, img_rgb.dtype)
        for y in range(img_rgb.shape[0]):
            for x in range(img_rgb.shape[1]):
                for c in range(img_rgb.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*img_rgb[y,x,c] + beta, 0, 255)
        #cv2.imshow("Original Image", img_rgb)
        #cv2.waitKey(100)
        #cv2.imshow("New Image", new_image)
        #cv2.waitKey(100)
        return new_image                

    def cleanup(self, img):
        new_image = np.zeros(img.shape, img.dtype)
    
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    if (img[y, x, c] > 80):
                        new_image[y, x, c] = 255
                    elif (img[y, x, c] < 20):
                        new_image[y, x, c] = 0
                    else:
                        new_image[y, x, c] = img[y, x, c]  
        return new_image

    def resizeToCard(self, img_final):
        return cv2.resize(img_final, (685, 450))

tmp_canvas = Cartoonizer()
raw = cv2.VideoCapture('Horses_2.mp4')
ret,frame = raw.read()
frame_count = 0
height, width, layers = frame.shape
video = cv2.VideoWriter("Horses_2.avi", 0, 10, (width,height))
images = []

while frame_count<30:
    frame_count = frame_count + 1
    print(frame_count)
    res = tmp_canvas.render(frame)
    images.append(res)
    video.write(res)
    ret,frame = raw.read()

video.release()

imageio.mimsave('horse.gif',images)