from pystackreg import StackReg
from skimage.filters import threshold_minimum, threshold_li
import scipy
import mahotas as mh
import math
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, hough_line
from skimage.morphology import skeletonize
import skimage
import matplotlib.pyplot as plt

def correct_drift(reference, move):
    sr = StackReg(StackReg.RIGID_BODY)
    
    transformation_matrix = sr.register(reference, move)
    
    out_rot = sr.transform(move, transformation_matrix)
    return out_rot, transformation_matrix

def get_img_half(image, t_b):
    height = int(image.shape[0]/2)
    if t_b == "top":
        return image[:height,:]
    if t_b == "bottom":
        return image[height:,:]
    
def get_orientation(image, debug = False):
    bin_image = (image > threshold_li(image)) * 1
    dilated = scipy.ndimage.morphology.binary_dilation(bin_image, iterations = 30)
    labeled, num_regions  = mh.label(dilated)
    sizes = mh.labeled.labeled_size(labeled)
    mh.labeled.labeled_size(labeled)

    if len(sizes) > 2:
        too_small = np.where(sizes < sizes[1]-1)
        labeled = mh.labeled.remove_regions(labeled, too_small)
    skeleton = skeletonize(labeled)
    h, theta, d = hough_line(skeleton)
    origin = np.array((0, skeleton.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    gradient = (y0-y1) /origin[1]
    if debug == False:
        return math.degrees(np.arctan(gradient))
    elif debug == True:
        f, ax = plt.subplots(figsize=(40,20))
        plt.imshow(bin_image)
        plt.show()
        f, ax = plt.subplots(figsize=(40,20))
        plt.imshow(dilated)
        plt.show()
        f, ax = plt.subplots(figsize=(40,20))
        plt.imshow(skeleton)
        plt.show()
        return math.degrees(np.arctan(gradient))
def fix_orientation(image, angle):
    return skimage.transform.rotate(image = image, angle = -1*angle)