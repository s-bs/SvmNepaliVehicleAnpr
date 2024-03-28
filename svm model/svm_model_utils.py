import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from joblib import dump, load
from skimage.color import rgb2gray

class SVMmodel:
    def data_prepare(self,dir,categories):
    # Initialize empty lists for features and labels
        features = []
        labels = []
        # preparing a dataset for training a machine learning model
        # Loop through each category
        for category in categories:
            path = os.path.join(dir, category)
            class_num = categories.index(category)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = imread(img_path, as_gray=True)
                resized_image = resize(image, (28, 28))
                fd = hog(resized_image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
                features.append(fd)
                labels.append(class_num)
        return features,labels
    
    def train_model(self,features,labels):
        # Convert features and labels to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create an SVM classifier
        clf = svm.SVC()

        # Train the SVM classifier
        clf.fit(X_train, y_train)

        # Save the trained model for future use
        model_path = "D:/Python/nepali-license-plate-number-detection-SVM/resources/model/svm_model.joblib"
        dump(clf, model_path)

        # Make predictions on the testing set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)
        # return accuracy
        
    def category_predict(self,roi, clf_loaded, categories):
        roi_gray = rgb2gray(roi)
        resized_image = resize(roi_gray, (28, 28))
        fd = hog(resized_image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        X = np.array([fd])
        # Predict the label of the image
        prediction = clf_loaded.predict(X)
        # Get the predicted category
        predicted_category = categories[prediction[0]]
        return predicted_category