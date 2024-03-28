import cv2
import os
from PIL import Image
import numpy as np
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew

class Numberplatecharacter:
    def extract_red_threshold(self,image):
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)
        full_mask = lower_mask + upper_mask;
        result = cv2.bitwise_and(result, result, mask=full_mask)
        return full_mask,result
    
    def extract_number_plate(self, maskimage,image):
        # Find contours in the binary mask image
        contours, _ = cv2.findContours(maskimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a copy of the original image
        image_with_rectangles = image.copy()
        # Draw around the contours
        for contour in contours:
            if cv2.contourArea(contour) > 30000:
                image_with_rectangles = cv2.drawContours(image_with_rectangles, [contour], -1, (0, 255, 0), 2)
                # Get the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Crop the original image using the bounding rectangle
                region_inside_contour = image[y:y+h, x:x+w]
                # Return the region inside the contour
                return region_inside_contour
            
    def find_word(self, inverted_mask,localized_img):
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segement_character = localized_img.copy()
        roi_list = []  # List to store the ROIs
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(x)
            if h < 100: 
                if cv2.contourArea(contour) > 400 and cv2.contourArea(contour) < 4500 :
                    # segement_character_rect_retc = cv2.rectangle(segement_character, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi = segement_character[y:y+h, x:x+w]
                    roi_list.append(roi)  # Append ROI to the list
        return roi_list
