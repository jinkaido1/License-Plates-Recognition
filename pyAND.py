import cv2
import numpy as np
import sys
import glob
import math
import time
import os
import pytesseract
import tesserocr
from PIL import Image


def validate_contour(contour, img, aspect_ratio_range, area_range):
    rect = cv2.minAreaRect(contour)
    img_width = img.shape[1]
    img_height = img.shape[0]
    box = cv2.boxPoints(rect) 
    box = np.int0(box)

    X = rect[0][0]
    Y = rect[0][1]
    angle = rect[2] 
    width = rect[1][0]
    height = rect[1][1]

    angle = (angle + 180) if width < height else (angle + 90)

    output=False

    if (width > 0 and height > 0) and ((width < img_width/2.0) and (height < img_width/2.0)):
        aspect_ratio = float(width)/height if width > height else float(height)/width
        if aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]:

        	if((height*width > area_range[0]) and (height*width < area_range[1])):

        		box_copy = list(box)
        		point = box_copy[0]
        		del(box_copy[0])
        		dists = [((p[0]-point[0])**2 + (p[1]-point[1])**2) for p in box_copy]
        		sorted_dists = sorted(dists)
        		opposite_point = box_copy[dists.index(sorted_dists[1])]
        		tmp_angle = 90

        		if abs(point[0]-opposite_point[0]) > 0:
        			tmp_angle = abs(float(point[1]-opposite_point[1]))/abs(point[0]-opposite_point[0])
        			tmp_angle = rad_to_deg(math.atan(tmp_angle))

        		if tmp_angle <= 45:
        			output = True
    return output

def deg_to_rad(angle):
	return angle*np.pi/180.0

def rad_to_deg(angle):
	return angle*180/np.pi

def enhance(img):
	kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
	return cv2.filter2D(img, -1, kernel)

def process_image(name, debug, **options):

	se_shape = (16,4)

	if options.get('type') == 'rect':
		se_shape = (17,4)

	elif options.get('type') == 'square':
		se_shape = (7,6)

	raw_image = cv2.imread(name,1)
	input_image = raw_image

	gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
	gray = enhance(gray)
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	gray = cv2.Sobel(gray, -1, 1, 0)
	_, sobel = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	se = cv2.getStructuringElement(cv2.MORPH_RECT, se_shape)
	gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)
	ed_img = gray

	_, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	print(contours)

	#font = cv2.FONT_HERSHEY_SIMPLEX

	for contour in contours:
		aspect_ratio_range = (2.2, 12)
		area_range = (500, 18000)

		if options.get('type') == 'rect':
			aspect_ratio_range = (2.2, 12)
			area_range = (500, 18000)

		elif options.get('type') == 'square':
			aspect_ratio_range = (1, 2)
			area_range = (300, 8000)

		if validate_contour(contour, gray, aspect_ratio_range, area_range):
			rect = cv2.minAreaRect(contour)  # Returns ( center (x,y), (width, height), angle of rotation )
			box = cv2.boxPoints(rect) 
			box = np.int0(box)  
			Xs = [i[0] for i in box]
			Ys = [i[1] for i in box]
			x1 = min(Xs)
			x2 = max(Xs)
			y1 = min(Ys)
			y2 = max(Ys)

			angle = rect[2]
			if angle < -45:
				angle += 90 

			W = rect[1][0]
			H = rect[1][1]
			aspect_ratio = float(W)/H if W > H else float(H)/W

			center = ((x1+x2)/2,(y1+y2)/2)
			size = (x2-x1, y2-y1)
			M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
			tmp = cv2.getRectSubPix(ed_img, size, center)
			tmp = cv2.warpAffine(tmp, M, size)
			TmpW = H if H > W else W
			TmpH = H if H < W else W
			tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))
			_,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			white_pixels = 0

			for x in range(tmp.shape[0]):
				for y in range(tmp.shape[1]):
					if tmp[x][y] == 255:
						white_pixels += 1

			edge_density = float(white_pixels)/(tmp.shape[0]*tmp.shape[1])

			tmp = cv2.getRectSubPix(raw_image, size, center)
			tmp = cv2.warpAffine(tmp, M, size)
			TmpW = H if H > W else W
			TmpH = H if H < W else W
			tmp = cv2.getRectSubPix(tmp, (int(TmpW), int(TmpH)), (size[0]/2, size[1]/2))

			if edge_density > 0.5:
				#detection_image = cv2.drawContours(input_image, [box], 0, (127,0,255),2)
				# cv2.imshow('det', input_image)
				# cv2.waitKey(0)

				bounding_box = cv2.boundingRect(np.int32(box)) # cv2.boxPoints
				crop_x = bounding_box[0]
				crop_y = bounding_box[1]
				crop_w = bounding_box[2]
				crop_h = bounding_box[3]
				print(box, bounding_box)
				plate = raw_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
				M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
				plate = cv2.getRectSubPix(raw_image, size, center)
				plate = cv2.warpAffine(plate, M, size)
				#cv2.destroyAllWindows()

				#cv2.imshow('Crop', plate)

			# Cropping
			# bounding_box = cv2.boundingRect(np.int32(box))
			# crop_x = bounding_box[0]
			# crop_y = bounding_box[1]
			# crop_w = bounding_box[2]
			# crop_h = bounding_box[3]

			# print(box, bounding_box)
			# plate = input_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
			# cv2.imshow('Crop', plate)

	return input_image, plate


if len(sys.argv) < 2:
	print('usage:\n python pyANPD.py <image_file_path>')
	exit(0)

path = sys.argv[1]

t1 = time.time()
o1, plate = process_image(path, 0, type='rect')
cv2.imshow('Detection', o1)
cv2.imshow('Plate', plate)
#cv2.imwrite('%s-detected.png' % path[:path.rfind('.')], o1)
cv2.waitKey(0)

print('Time taken: %d ms'%((time.time()-t1)*1000))

# Recognition
cv2.destroyAllWindows()
print("Without preprocessing: ")
cv2.imshow('Plate', plate)
print("Pytesseract: {}".format(pytesseract.image_to_string(plate)))
img = Image.fromarray(plate)
print("OCR: {}".format(tesserocr.image_to_text(img)))  # print ocr text from image

print("With preprocessing: ")
image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#image = cv2.resize(image, (640, -1), interpolation=cv2.INTER_CUBIC)
#image = cv2.resize(image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # INTER_AREA to decrease
image = cv2.bilateralFilter(image, 11, 17, 17)
image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('Processed Plate', image)
print("Pytesseract: {}".format(pytesseract.image_to_string(image)))
img = Image.fromarray(image)
print("OCR: {}".format(tesserocr.image_to_text(img)))
cv2.waitKey(0)

