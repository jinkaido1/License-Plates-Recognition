import numpy as np
import cv2

img = cv2.imread("images/lebron_james.jpg")
height = img.shape[0]
width = img.shape[1]

mask = np.zeros((height, width), dtype=np.uint8)
points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
cv2.fillPoly(mask, points, (255))

res = cv2.bitwise_and(img,img,mask = mask)

rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
print(cropped)

cv2.imshow("cropped" , cropped )
cv2.imshow("same size" , res)

print(cropped.shape[0], cropped.shape[1])

print(points.max)
M = cv2.getRotationMatrix2D((cropped.shape[0]/2, cropped.shape[1]/2),90,1)
dst = cv2.warpAffine(cropped,M,(cropped.shape[0],cropped.shape[1]))
cv2.imshow("Test" , dst)

M = cv2.getRotationMatrix2D((height/2, width/2),45,1)
dst = cv2.warpAffine(img,M,(height,width))
cv2.imshow("Test2" , dst)
cv2.imwrite('ttt.jpg', dst)
cv2.waitKey(0)