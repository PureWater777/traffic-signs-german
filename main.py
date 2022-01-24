import csv

import cv2
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=1000)

# Importing Images

path = "ImageQuery"
path_test = "Test"
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
    desList = []
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
        try:
            matches = bf.knnMatch(des, des2, k=2)
        except cv2.error as err:
            print(err)
            matchList.append(0)
            continue
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
        except ValueError as err:
            print(err)
            continue
        matchList.append(len(good))

    return matchList



#Case 1
# img2 = cv2.imread('Test/00001.ppm')
# result = findID(img2, desList)
# print(result)

# #Case 2
# img2 = cv2.imread('Test/00009.ppm')
# result = findID(img2, desList)
# print(result)

#Case 3
img2 = cv2.imread('Test/00006.ppm')
result = findID(img2, desList)
print(result)

# dataset = 'Test'
# signs = {}
# csv_file_path = os.path.join(dataset, 'index.csv')
# with open(csv_file_path, mode='r') as csv_file:
#     counter = 0
#     file_len = 12631
#     next(csv_file)
#     for row in csv_file:
#         counter += 1
#         try:
#             row = row.split(",")
#             img_id = row[7]
#             img_id_replaced = img_id.replace('.png', '.ppm')
#             print(f"{str(counter)}/{str(file_len)}", end="\r")
#             print(img_id_replaced)
#             img_path = os.path.join(dataset, img_id_replaced.rstrip("\n"))
#             img = cv2.imread(img_path)
#             class_id = row[6]
#             img_prediction_list = findID(img, desList)
#             if not all(img_prediction_list):
#                 class_id = "Undefined"
#             img_prediction_index = img_prediction_list.index(max(img_prediction_list))
#         except TypeError as err:
#             print(err)
#             print(img_id_replaced)
#
#         try:
#             result_table = signs[str(class_id)]
#         except KeyError:
#             signs.update({
#                 str(class_id): {
#                     'ok': 0,
#                     'not': 0,
#                 }
#             })
#             result_table = signs[str(class_id)]
#         key = 'not'
#         if img_prediction_index == class_id:
#             key = 'ok'
#         how_much = result_table[key] + 1
#         result_table[key] = how_much
#
# with open("german_results.csv", "w") as result_csv:
#     csv_writer = csv.writer(result_csv, delimiter=",")
#     csv_writer.writerow(["ClassID", "Ok", "NOT"])
#     for sign_id, results in signs.items():
#         csv_writer.writerow([str(sign_id), results["ok"], results['not']])
