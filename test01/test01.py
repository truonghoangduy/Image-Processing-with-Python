import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
# import skimage
from skimage import filters

# Load Image
image = cv2.imread('picture/05.jpg')
#Prewiit Kernal  
# // Could be use on cv2.filter2D :
#  Aka 2 kernal 3x3 kernal
kernelx3 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely3 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) 
kernalx5 = np.array([
[2,2,4,2,2],
[1,1,2,1,1],
[0,0,0,0,0],
[-1,-1,-2,-1,-1],
[-2,-2,-4,-2,-2]])
kernaly5 = np.array([[2,1,0,-1,-2],[2,1,0,-1,-2],[4,2,0,-2,-4],[2,1,0,-1,-2],[2,1,0,-1,-2]])

def showImage(name,image,x,y):
     # show image with opencv
     cv2.namedWindow(name,cv2.WINDOW_NORMAL)
     cv2.resizeWindow(name, 400,400)
     cv2.moveWindow(name,x,y)
     cv2.imshow(name,image)

def rbgHistogram(image):
     color = ('b','g','r')
     for i,col in enumerate(color):
          histr = cv2.calcHist([image],[i],None,[256],[0,256])
          plt.plot(histr,color = col)
          plt.xlim([0,256])

def prewittKernal(name,image,kernelx,kernely,x,y):
     imageX = cv2.filter2D(image,-1,kernelx)
     imageY =cv2.filter2D(image,-1,kernely)
     # Origanal to megre two image after filter 
     # atan22 = cv2.fastAtan2(imageX.flatten(),imageY.flatten())
     # prewittFlitter= cv2.addWeighted(imageX,0.5,imageY,0.5,0)
     prewittFlitter= cv2.add(imageX,imageY)
     showImage(name,prewittFlitter,x,y)     

# Gray scale
grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Blur Prewitt with skiImage
blurPrewitt = filters.prewitt(image)
# BW image thresh hold 127 from gray scle imgae
(thresh,bWimage) = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
print(thresh)
# Call function For show Image
showImage("Orgianal",image,0,0)
showImage("Prewitt Blur",blurPrewitt,400,0)
showImage("GrayImage",grayImage,800,0)
showImage("BW",bWimage,1200,0)
rbgHistogram(image)

# Call function For Prewwit DIY kernal show Image

prewittKernal("Prewitt3x3",grayImage,kernelx3,kernely3,1400,400)
prewittKernal("Prewitt5x5",grayImage,kernalx5,kernaly5,1700,400)
plt.hist(grayImage.ravel(),256,[0,255],label="Gray Color Distibution")
plt.show()





key = cv2.waitKey(0)
if key == 13:
     plt.close()
     cv2.destroyAllWindows()

