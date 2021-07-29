import cv2
import numpy as np

class Cartoonizer:
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
        
        img_color_downsampled = self.enhanceContrast(img_color_downsampled)
        cv2.imwrite("enhanced.jpg", img_color_downsampled)        
        
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
        img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_screentone = self.screentone(img_gray_rgb)
        cv2.imwrite("screentoned.jpg", img_screentone)
        
        img_gray_blurred = cv2.medianBlur(img_gray, 3)
        cv2.imwrite("blurred.jpg", img_gray_blurred)
        
        #6 Detect and enhance edges:
        
        img_edge = cv2.adaptiveThreshold(
            img_gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 3)
        (x, y, c) = img_color_upsampled.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)        
        cv2.imwrite("edge.jpg", img_edge)
        
        #7 Add edge to color image:
        img_edge = cv2.fastNlMeansDenoisingColored(img_edge, None, 120, 10, 7, 25)
        return cv2.bitwise_and(img_color_upsampled, img_edge)
        
        
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
    
    def enhanceContrast(self, img_rgb):
        alpha = 1.5
        beta = -20
        new_image = np.zeros(img_rgb.shape, img_rgb.dtype)
        for x in range(img_rgb.shape[0]):
            for y in range(img_rgb.shape[1]):
                for c in range(img_rgb.shape[2]):
                    new_image[x, y, c] = np.clip(alpha * img_rgb[x, y, c] + beta, 0, 255)
        return new_image
    
    def screentone(self, img_grey):
        st1 = cv2.imread("screentone1.jpg")
        st2 = cv2.imread("screentone2.jpg")
        st3 = cv2.imread("screentone3.jpg")
        new_image = np.zeros(img_grey.shape, img_grey.dtype)
        for x in range(img_grey.shape[0]):
            for y in range(img_grey.shape[1]):
                for c in range(img_grey.shape[2]):
                    if (img_grey[x, y, c] > 150 and img_grey[x, y, c] < 180):
                        new_image[x, y, c] = st1[x, y, c]
                    elif (img_grey[x, y, c] > 80 and img_grey[x, y, c] < 100):
                        new_image[x, y, c] = st2[x, y, c]
                    elif (img_grey[x, y, c] > 30 and img_grey[x, y, c] < 60):
                        new_image[x, y, c] = st3[x, y, c]     
                    else:
                        new_image[x, y, c] = img_grey[x, y, c]
        return new_image

canvas = Cartoonizer()
res = canvas.render("musk.jpg")
cv2.imwrite("final.jpg", res)

blackimg = cv2.imread("edge.jpg")
dst = cv2.fastNlMeansDenoisingColored(blackimg, None, 120, 10, 7, 25)
cv2.imwrite("denoised.jpg", dst)