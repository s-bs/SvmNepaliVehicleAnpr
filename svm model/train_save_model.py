import sys
sys.path.append(r'D:\Python\nepali-license-plate-number-detection-SVM\svm model')
import svm_model_utils

# Prepare data and train and train save model
# Set the directory path and categories
svm_model_obj = svm_model_utils.SVMmodel()
dir = "D:/Python/nepali-license-plate-number-detection-SVM/resources/images_dataset/images"
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'pa']
# preparing a dataset for training a machine learning model
features,labels=svm_model_obj.data_prepare(dir,categories)
# train and save model
train_save_model = svm_model_obj.train_model(features,labels)