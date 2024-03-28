import cv2
import os
import matplotlib.pyplot as plt
from joblib import dump, load
import sys
sys.path.append(r'D:\Python\nepali-license-plate-number-detection-SVM\service\number_plate_character')
sys.path.append(r'D:\Python\nepali-license-plate-number-detection-SVM\svm model')
import svm_model_utils
import number_plate_character_utils


director_path = "D:/Python/nepali-license-plate-number-detection-SVM/resources"
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'pa']
clf_loaded = load("D:/Python/nepali-license-plate-number-detection-SVM/resources/model/svm_model.joblib")
char_obj = number_plate_character_utils.Numberplatecharacter()
predict_obj = svm_model_utils.SVMmodel()
for filename in os.listdir(director_path):
  if filename.endswith(".jpg"):
    image_path = os.path.join(director_path, filename)
    image = cv2.imread(image_path)
    # original HSV based thresholding for red color
    full_mask,result = char_obj.extract_red_threshold(image)
    # Localized license plate
    localized_img = char_obj.extract_number_plate(full_mask, image)
    # localized HSV based thresholding for red color
    mask_img,result_img= char_obj.extract_red_threshold(localized_img)
    inverted_mask_img = cv2.bitwise_not(mask_img)
    # segment word
    roi_list = char_obj.find_word(inverted_mask_img, localized_img)
    # Display each ROI
    result_num = []
    for roi in roi_list:
        reconized_text = predict_obj.category_predict(roi,clf_loaded, categories)
        result_num.append(reconized_text)
        
    print("Vehical Number",result_num)