# Main.py

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate
import pytesseract
import re
import matplotlib.pyplot as plt
from imutils import contours
pytesseract.pytesseract.tesseract_cmd=r"c:\Program Files\Tesseract-OCR\tesseract.exe"



# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    #imgOriginalScene  = cv2.imread("LicPlateImages/1.png")               # open image
    imgOriginalScene = cv2.imread("plate436.jpg")  # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        #cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
        #cv2.imshow("imgThresh", licPlate.imgThresh)
        #gray = cv2.resize(imgOriginalScene, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(imgOriginalScene, (5, 5), 0)
        bitwise = cv2.bitwise_not(blur)
        cv2.imshow("not", bitwise)
        image = cv2.imread('plate436.jpg')
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")
        plate = ""
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            center_y = y + h / 2
            if area > 3000 and (w > h) and center_y > height / 2:
                ROI = image[y:y + h, x:x + w]
                cv2.imshow(ROI)
                data = pytesseract.image_to_string(ROI, lang='fra+ara', config='--psm 10')
                plate += data
        print('License plate:', plate)
        print(plate_string(bitwise))



    # end if else

    cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main

###################################################################################################
# tesseract
def plate_string(img):
    # extract all strings from the image
    custom_config = r'-l fra+ara --psm 10'
    text = pytesseract.image_to_string(img, config=custom_config)
    x = text.split('\n')

    # clean the string
    return x[0]



###################################################################################################
if __name__ == "__main__":
    main()


















