# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:35:39 2020

@author: deniz.kilic
"""

import sys
import os
import cv2
from math import floor, ceil, log
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from skimage.color import rgb2gray
from skimage.transform import resize

# Celik, T. (2009). Unsupervised change detection in satellite images using principal component analysis
# and k-means clustering. IEEE Geoscience and Remote Sensing Letters, 6(4), 772–776. https://doi.org/10.1109/LGRS.2009.2025059

# Compute log
def logTransform(c, f):
    g = c * log(float(1 + f), 10)
    return g


# Apply logarithmic transformation for an image
def logTransformImage(image, outputMax=255, inputMax=255):
    c = outputMax / log(inputMax + 1, 10)  # for values 255 it gives approximately 105

    # Read pixels and apply logarithmic transformation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Get pixel value at (x,y) position of the image
            f = image[i, j]

            # Do log transformation of the pixel
            image[i, j] = round(logTransform(c, f))

    return image


def find_vector_set(diff_image, new_size, bs):
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / (bs * bs)), (bs * bs)))

    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block = diff_image[j:j + bs, k:k + bs]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + bs
            j = j + bs
        i = i + 1

    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec

    return vector_set, mean_vec


def find_FVS(EVS, diff_image, mean_vec, new, bs):
    i = floor(bs / 2)
    feature_vector_set = []

    while i < new[0] - floor(bs / 2):
        j = floor(bs / 2)
        while j < new[1] - floor(bs / 2):
            block = diff_image[i - floor(bs / 2):i + ceil(bs / 2), j - floor(bs / 2):j + ceil(bs / 2)]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j + 1
        i = i + 1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    return FVS


def clustering(FVS, components, new, bs):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)

    least_index = min(count, key=count.get)
    if bs % 2 == 0:
        change_map = np.reshape(output, (new[0] - bs, new[1] - bs))
    else:
        change_map = np.reshape(output, (new[0] - (bs - 1), new[1] - (bs - 1)))

    return least_index, change_map

def isGray(img):
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False

def prepareOutputImage(image, _map):
    if len(image.shape) == 2:
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    w, h, c = image.shape

    n_size = (h, w)
    _map = cv2.resize(_map, n_size)
    image[:, :, 2] += _map
    return image

def find_PCAKmeans(imagepath1, imagepath2, bs, components, outputType, displayType):
    print('Operating the change detection...')

    image1 = cv2.imread(imagepath1)
    image2 = cv2.imread(imagepath2)

    if isGray(image1):
        image1 = image1[:, :, 0]

    if isGray(image2):
        image2 = image2[:, :, 0]

    new_size = np.asarray(image1.shape) / bs
    new_size = new_size.astype(int) * bs

    image1 = resize(image1, (new_size[0], new_size[1]), preserve_range=True).astype(np.int16)
    image2 = resize(image2, (new_size[0], new_size[1]), preserve_range=True).astype(np.int16)

    ######################### Choose difference type of 2 image ###########################
    # https://pythontic.com/image-processing/pillow/logarithmic%20transformation
    # comment out in order to choose abs(diff) or log_ratio
    # in practice we suppose that it may work better with ratio

    image2[image2 == 0] = 1
    ratio_image = (image1 / image2)
    ratio_image = rgb2gray(ratio_image)
    logRatio_image = logTransformImage(ratio_image)
    diff_image = logRatio_image

    print('\ncomputing PCA ', new_size)

    vector_set, mean_vec = find_vector_set(diff_image, new_size, bs)

    pca = PCA()
    pca.fit(vector_set)
    EVS = pca.components_

    FVS = find_FVS(EVS, diff_image, mean_vec, new_size, bs)

    print('\ncomputing k means')

    least_index, change_map = clustering(FVS, components, new_size, bs)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map = change_map.astype(np.uint8)

    if outputType == 'clean':
        kernel = np.asarray(((0, 0, 1, 0, 0),
                             (0, 1, 1, 1, 0),
                             (1, 1, 1, 1, 1),
                             (0, 1, 1, 1, 0),
                             (0, 0, 1, 0, 0)), dtype=np.uint8)
        _map = cv2.erode(change_map, kernel)
    else:
        _map = change_map

    if displayType == '' or displayType == 'none':
        return _map
    else:
        if displayType == 'first':
            return prepareOutputImage(image1, _map)
        if displayType == 'second':
            return prepareOutputImage(image2, _map)

if __name__ == '__main__':

    args = sys.argv[1:]
    inputCheck = True
    args_len = len(args)

    if args_len != 6 and args_len != 7:
        print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
        print('You give wrong amount of input arguments')
        inputCheck = False
    else:
        image_path1 = args[0]
        image_path2 = args[1]
        if inputCheck and not (os.path.isfile(image_path1) and os.path.isfile(image_path2)):
            print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
            print(image_path1 + ' and/or ' + image_path2 + ' is/are not (a) existing file(s).')
            inputCheck = False

        output_loc = args[2]
        output_dir = os.path.dirname(output_loc)
        if inputCheck and not os.path.isdir(output_dir):
            print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
            print(output_dir + ' is not a existing directory.')
            inputCheck = False

        output_type = args[3]
        if inputCheck and not (output_type == 'original' or output_type == 'clean'):
            print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
            print('outputType can be \'clean\' or \'original\'')
            inputCheck = False

        display_type = ''
        ind = 4
        if args_len == 7:
            ind = ind + 1
            display_type = args[4]
            display_types = ['first', 'second', 'none']
            if inputCheck and not (display_type in display_types):
                print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
                print('Undefined visualization type. Valid visualization types: \'first\', \'second\' ,\'none\'')
                inputCheck = False

        bs = args[ind]
        if inputCheck and not str.isdigit(bs):
            print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
            print(bs + ' is not an integer.')
            inputCheck = False
        else:
            bs = int(bs)

        components = args[ind + 1]
        if inputCheck and not str.isdigit(components):
            print('Usage: PCAKmeans <imageFilePath1> <imageFilePath2> <outputLoc> <outputType> <displayType(optional)> <blockSize> <components>')
            print(components + ' is not an integer.')
            inputCheck = False
        else:
            components = int(components)

    if inputCheck:
        _map = find_PCAKmeans(image_path1, image_path2, bs, components, output_type, display_type)
        cv2.imwrite(output_loc, _map)
