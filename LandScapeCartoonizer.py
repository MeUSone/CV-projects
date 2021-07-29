import cv2
import numpy as np

class LandScapeCartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.
    """
    
    def __init__(self):
        pass
    
    def render(self, color_image_file):
        img_rgb = cv2.imread(color_image_file)
        
        #1 Down Sample:
        
        numDownSamples = 2
        img_color = img_rgb
        img_color_downsampled = self.downSample(img_color, numDownSamples)
        cv2.imwrite("downsampled.jpg", img_color_downsampled)
        
        #2 Enhance Contrast:
        img_color_enhanced = self.enhanceContrast(img_color_downsampled)
        cv2.imwrite("enhanced.jpg", img_color_enhanced)
        
        
        #3 Bilateral filter:
        
        numBilateralFilters = 50
        img_color_filtered = img_color_downsampled
        img_color_filtered = self.bilateralFilters(img_color_filtered, numBilateralFilters)
        cv2.imwrite("filtered.jpg", img_color_filtered)
        
        #4 Up Sample:
        
        img_color_upsampled = img_color_filtered
        img_color_upsampled = self.upSample(img_color_upsampled, numDownSamples)
        cv2.imwrite("upsampled.jpg", img_color_upsampled)
        
        #5 Convert to BW & Median Blur:
        
        img_gray = cv2.cvtColor(img_color_upsampled, cv2.COLOR_RGB2GRAY)
        img_gray_blurred = cv2.medianBlur(img_gray, 3)
        cv2.imwrite("blurred.jpg", img_gray_blurred)
        
        #6 detect and enhance edges
        
        img_edge = cv2.adaptiveThreshold(
            img_gray_blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY,9,3)
        (x,y,z) = img_color_upsampled.shape
        img_edge = cv2.resize(img_edge,(y,x))
        img_edge = cv2.cvtColor(img_edge,cv2.COLOR_GRAY2RGB) 
        img_edge = self.cleanUp(img_edge)
        cv2.imwrite("edge.jpg", img_edge)
        
        
        #7 Add edge to color image
        img_edge = cv2.fastNlMeansDenoisingColored(img_edge,None,120,10,7,25)
        return cv2.bitwise_and(img_color_upsampled, img_edge)
        
        
    #for _ in range(numDownSamples)
    def downSample(self, img_color, numDownSamples):
        while (numDownSamples > 0):
            numDownSamples -= 1
            img_color = cv2.pyrDown(img_color)
        return img_color
    
    def upSample(self, img_color, numDownSamples):
        while (numDownSamples > 0):
            numDownSamples -= 1
            img_color = cv2.pyrUp(img_color)
        return img_color    
    
    def bilateralFilters(self, img_color, numBilateralFilters):
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        return img_color
    
    def enhanceContrast(self,img_rgb):
        alpha = 1.5
        beta = -20
        new_image = np.zeros(img_rgb.shape, img_rgb.dtype)
        for x in range(img_rgb.shape[0]):
            for y in range(img_rgb.shape[1]):
                for z in range(img_rgb.shape[2]):
                    new_image[x,y,z] =  np.clip(alpha * img_rgb[x,y,z]+beta,0,255)
        return new_image
    
    def cleanUp(self,img_grey):
        new_image = np.zeros(img_grey.shape, img_grey.dtype)
        black_thresh = 40;
        white_thresh = 100;
        for x in range(img_grey.shape[0]):
            for y in range(img_grey.shape[1]):
                for z in range(img_grey.shape[2]):
                    if(img_grey[x,y,z]<black_thresh):
                        new_image[x,y,z] = 0
                    elif(img_grey[x,y,z]<white_thresh):
                        new_image[x,y,z] = 255
                    else:
                        new_image[x,y,z]=img_grey[x,y,z]
        return new_image        

canvas = LandScapeCartoonizer()
res = canvas.render("musk.jpg")
cv2.imwrite("final.jpg",res)

blackimg = cv2.imread("edge.jpg")
dst = cv2.fastNlMeansDenoisingColored(blackimg,None,120,10,7,25)
cv2.imwrite("denoised.jpg",dst)