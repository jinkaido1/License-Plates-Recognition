import cv2
import sys
import pytesseract
 
if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: python ocr_simple.py image.jpg')
    sys.exit(1)
   
  # Read image path from command line
  imPath = sys.argv[1]
     
  # Uncomment the line below to provide path to tesseract manually
  # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
  #pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR'
 
  # Define config parameters.
  # '-l eng'  for using the English language
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l eng --oem 1')
 
  # Read image from disk
  im = cv2.imread(imPath, cv2.IMREAD_COLOR)

  # Image preprocessing
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
 
  # Run tesseract OCR on image
  #text = pytesseract.image_to_string(im, config=config)
  text = pytesseract.image_to_string(im) # Problemas con la configuración
 
  # Print recognized text
  print(text)
  
  # Showing image
  cv2.imshow('Image', im)
  cv2.waitKey(0)
