from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from imutils import paths
import argparse
import cv2
import os
import skimage
from skimage import data,filters,io,exposure,feature
import numpy as np
import IPython
import PIL
import easydict

ap=argparse.ArgumentParser()
ap.add_argument("-t","--training", required=True,
    help= "path to the training images")
ap.add_argument("-e","--testing", required=True,
    help="blah")
args = vars(ap.parse_args())

bob = LocalBinaryPatterns(24,8)
data = []
labels = []

for imagePath in paths.list_images(args['training']):
    # image = io.imread(imagePath)
    # # stretch = exposure.rescale_intensity(image)
    # grey = skimage.color.rgb2gray(image)
    image = cv2.imread(imagePath)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = bob.describe(grey)
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

StandardScaler.transform(data)
model = LinearSVC(C=1000.0, random_state=42)
model.fit(data,labels)

for imagePath in paths.list_images(args['testing']):
    # image = io.imread(imagePath)
    # stretch = exposure.rescale_intensity(image)
    # grey = skimage.color.rgb2gray(stretch)
    image = cv2.imread(imagePath)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = bob.describe(grey)
    prediction = model.predict(hist.reshape(1,-1))

    cv2.putText(image,prediction[0],(10,30),cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0,0,255),3)
    cv2.imshow('Image',image)
    cv2.waitKey(0)