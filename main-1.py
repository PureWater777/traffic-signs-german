import cv2
import numpy as np
import os


orb = cv2.ORB_create(nfeatures=1000)

# Importing Images

path = "ImageQuery"
path_test = "ImageTrain"
images = []
className = []
myList = os.listdir(path)
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
print(className)

# Finding Descriptors

def findDes(images):
    desList=[]
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

desList = findDes(images)

# Find ID

def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matchList = []
    for des in desList:
        matches = bf.knnMatch(des, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        matchList.append(len(good))
    return(matchList)
#
# #Case1
# img2 = cv2.imread('ImageTrain/000_1_0001.png')
# img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# result = findID(img2, desList)
# print(result)
#


# #Case3
# img2 = cv2.imread('ImageTrain/000_1_0003.png')
# img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# result = findID(img2, desList)
# print(result)


# import os
# dir = "ImageTrain"
# for file in os.listdir(dir):
#     img = cv2.imread(f"ImageTrain/{file}")
#     findID(img, desList)









