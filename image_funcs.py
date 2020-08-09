## Maths libraries
import scipy
import math
import numpy as np
import peakutils

## Image processing libraries
import skimage
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.morphology import skeletonize
from skimage.util import img_as_uint
from skimage.filters import threshold_minimum, threshold_li
from skimage import io
import imageio
import cv2
import mahotas as mh
from pystackreg import StackReg

## Misc libraries
import matplotlib.pyplot as plt
import glob
from joblib import Parallel, delayed
from tqdm import tqdm  
import re
import os
import time
import gc


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
        too_small = np.where(sizes < np.flip(np.sort(sizes))[1])
        labeled = mh.labeled.remove_regions(labeled, too_small)
        labeled = (labeled / np.max(np.unique(labeled))).astype(np.int)
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

def get_FOVs(directory, channel, padding):
    channel = channel
    z = 0
    FOV = "xy{}".format(str(z).zfill(padding))
    while len(glob.glob(directory + "{}**{}.tif".format(FOV,channel))) > 0:
        FOV = "xy{}".format(str(z).zfill(padding))
        z += 1
    num_FOVs = z-1
    
    FOVs = []
    for x in range(num_FOVs):
        FOVs.append("xy{}".format(str(x).zfill(padding)))
    
    return FOVs

def get_image_list(directory,FOV, channel):
    images =  glob.glob(directory + "{}**{}.tif".format(FOV,channel)) 
    images.sort()
    return images

def drift_correct_images(image_list, bit_size = np.uint16):
    ref = io.imread(image_list[0])
    drift_corrected_images = Parallel(n_jobs=-1)(delayed(correct_drift)(ref, io.imread(image_list[i])) for i in tqdm(range(len(image_list))))
    all_images = [drift_corrected_images[x][0].astype(bit_size) for x in range(len(drift_corrected_images))]
    trans_matrices = [drift_corrected_images[x][1] for x in range(len(drift_corrected_images))]
    return all_images, trans_matrices


def image_splitter(images):
    top_half_images = []
    bottom_half_images = []
    for i in range(len(images)):
        top_half_images.append(get_img_half(images[i],"top"))
        bottom_half_images.append(get_img_half(images[i],"bottom"))
    return top_half_images, bottom_half_images

def make_diagnostic_directories(output_directory):

    diag_directories = [
    "diagnostics",
    "diagnostics/rotations",
    "diagnostics/top_split",
    "diagnostics/bottom_split",
    "diagnostics/trench_finding"]

    for direc in diag_directories:
        try:
            os.mkdir(output_directory + direc)
        except:
            pass


def save_image(image_directory, frame):
    cv2.imwrite(image_directory, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

def register_transform_save_FOV(directory, FOV, phase_channel, fluor_channels):
    img_directories = get_image_list(directory,FOV, phase_channel)
    images, trans_matrices = drift_correct_images(img_directories)
    Parallel(n_jobs=-1)(delayed(save_image)(directory, image) for directory, image in tqdm(zip(img_directories, images), total = len(img_directories), desc = "Writing phase contrast FOV {} to disk".format(FOV), leave = False))
    if len(fluor_channels) > 0:
        for fluor_channel in fluor_channels:
            img_directories = get_image_list(directory,FOV, fluor_channel)
            images = []
            for x in range(len(img_directories)):
                images.append(io.imread(img_directories[x]))
            sr = StackReg(StackReg.RIGID_BODY)
            images = Parallel(n_jobs=-1)(  delayed(  sr.transform  )(image, trans_matrix) for image, trans_matrix in tqdm(zip(images, trans_matrices), total = len(images), desc = "Transforming FL channel {} in FOV {}".format(fluor_channel, FOV), leave = False))
            for x in range(len(images)):
                images[x] = images[x].astype(np.uint16)
            Parallel(n_jobs=-1)(delayed(save_image)(directory, image) for directory, image in tqdm(zip(img_directories, images), total = len(images), desc = "Writing FL channel {} images in FOV {} to disk".format(fluor_channel, FOV), leave = False))


def get_trenches(image, rotation, FOV, output_directory, top_bottom = None, min_dist = 20, thres = 1.4, top_thres_multiplier = 1, bottom_thres_multiplier = 2):

    test_image = img_as_uint(skimage.transform.rotate(image, rotation))
    bin_image = test_image > threshold_li(test_image) * 1

    cropped_bin = bin_image[int(bin_image.shape[0]*0.4):,]
    
    y_mean_intensity = np.mean(cropped_bin, axis=1)


    top_threshold = np.mean(y_mean_intensity)/top_thres_multiplier
    bottom_threshold = np.max(y_mean_intensity)/bottom_thres_multiplier
    top_threshold_line = np.argmax(y_mean_intensity > top_threshold) - 10
    bottom_threshold_line = np.argmax(y_mean_intensity > bottom_threshold)-10

    x_mean_intensity = np.mean(bin_image[top_threshold_line:bottom_threshold_line], axis=0)
    
    indexes = peakutils.indexes(x_mean_intensity, thres=thres*np.mean(x_mean_intensity), min_dist=min_dist)


    midpoints = (indexes[1:] + indexes[:-1]) / 2
    #f, ax = plt.subplots(figsize=(10,5))
    #plt.plot(x_mean_intensity)
    #plt.plot(y_mean_intensity)
    #plt.scatter(indexes, x_mean_intensity[indexes])

    f, ax = plt.subplots(figsize=(40,20))
    plt.imshow(test_image)
    plt.vlines(midpoints, ymin = 0, ymax = test_image.shape[0], color="r")
    plt.hlines(top_threshold_line, xmin = 0, xmax = test_image.shape[1], color="r")
    plt.hlines(bottom_threshold_line, xmin = 0, xmax = test_image.shape[1], color="r")
    plt.xlim(test_image.shape[1],0)
    plt.ylim(test_image.shape[0],0)
    plt.axis("off")
    if top_bottom == None:
        plt.savefig(output_directory +  "diagnostics/trench_finding/{}.jpeg".format(FOV), bbox_inches="tight")
        plt.close("all")
    else:
        plt.savefig(output_directory + "diagnostics/trench_finding/{}_{}.jpeg".format(FOV, top_bottom), bbox_inches="tight")
        plt.close("all")

    return top_threshold_line, bottom_threshold_line, midpoints

def extract_and_save_kymographs(FOV, output_directory, phase_channel, other_channels):
    trench_directory = output_directory + "trenches/" + FOV + "/" + phase_channel + "/"
    trench_directories = []
    _ = os.listdir(trench_directory)
    for x in range(len(_)):
        trench_directories.append(trench_directory + _[x] + "/")
    trench_directories.sort()
    for z in range(len(trench_directories)):
        trench_image_dirs = []
        _ = os.listdir(trench_directories[z])
        if len(_) == 0:
            pass
        else:
            for x in range(len(_)):
                trench_image_dirs.append(trench_directories[z] + _[x])
            trench_image_dirs.sort()

            trench_image_arrays = []
            for x in range(len(trench_image_dirs)):
                trench_image_arrays.append(io.imread(trench_image_dirs[x]))
            kymograph = np.concatenate(trench_image_arrays,axis=1)
            trench_ID = re.findall("trench_(\d+)", trench_image_dirs[0])[0]
            save_dir = output_directory + "kymographs/" + FOV + "/" + phase_channel + "/trench_{}.png".format(trench_ID)
            imageio.imwrite(save_dir,kymograph)
            
    for channel in other_channels:
        trench_directory = output_directory + "trenches/" + FOV + "/" + channel + "/"
        trench_directories = []
        _ = os.listdir(trench_directory)
        for x in range(len(_)):
            trench_directories.append(trench_directory + _[x] + "/")
        trench_directories.sort()
        for z in range(len(trench_directories)):
            trench_image_dirs = []
            _ = os.listdir(trench_directories[z])
            if len(_) == 0:
                pass
            else:
                for x in range(len(_)):
                    trench_image_dirs.append(trench_directories[z] + _[x])
                trench_image_dirs.sort()

                trench_image_arrays = []
                for x in range(len(trench_image_dirs)):
                    trench_image_arrays.append(io.imread(trench_image_dirs[x]))
                kymograph = np.concatenate(trench_image_arrays,axis=1)
                trench_ID = re.findall("trench_(\d+)", trench_image_dirs[0])[0]
                save_dir = output_directory + "kymographs/" + FOV + "/" + channel + "/trench_{}.png".format(trench_ID)
                imageio.imwrite(save_dir,kymograph)


def extract_and_save_trenches(FOV, rotation, double_mm, directory, output_directory, phase_channel, other_channels, trench_positions):
    FOV_timepoint_images = get_image_list(directory,FOV, phase_channel)
    if double_mm == 1:
        for t in range(len(FOV_timepoint_images)):
            timepoint = re.findall("T(\d+)", FOV_timepoint_images[t])[0]
            image_dir = [image for image in FOV_timepoint_images if timepoint in image][0]
            image = io.imread(image_dir)
            image = img_as_uint(skimage.transform.rotate(image, rotation))
            top_half, bottom_half = get_img_half(image,"top"), get_img_half(image,"bottom")
            bottom_half = img_as_uint(skimage.transform.rotate(bottom_half, 180))
            ## Do the top split of the FOV first
            top_threshold_line = trench_positions["{}_top".format(FOV)][0]
            bottom_threshold_line = trench_positions["{}_top".format(FOV)][1]
            midpoints = trench_positions["{}_top".format(FOV)][2]
            for y in range(len(midpoints)-1):
                save_dir = output_directory + "trenches/" + FOV + "_top/" + phase_channel + "/" + "trench_{}".format(str(y).zfill(2)) + "/T{}".format(timepoint) + ".png"
                cropped_top = top_half[top_threshold_line:bottom_threshold_line,int(midpoints[y]):int(midpoints[y+1])]
                imageio.imwrite(save_dir,cropped_top)
            ## Do the bottom split of the FOV next
            top_threshold_line = trench_positions["{}_bottom".format(FOV)][0]
            bottom_threshold_line = trench_positions["{}_bottom".format(FOV)][1]
            midpoints = trench_positions["{}_bottom".format(FOV)][2]
            for y in range(len(midpoints)-1):
                save_dir = output_directory + "trenches/" + FOV + "_bottom/" + phase_channel + "/" + "trench_{}".format(str(y).zfill(2)) + "/T{}".format(timepoint) + ".png"
                cropped_top = bottom_half[top_threshold_line:bottom_threshold_line,int(midpoints[y]):int(midpoints[y+1])]
                imageio.imwrite(save_dir,cropped_top)
        for channel in other_channels:
            FOV_timepoint_images = get_image_list(directory,FOV, channel)
            for t in range(len(FOV_timepoint_images)):
                timepoint = re.findall("T(\d+)", FOV_timepoint_images[t])[0]
                image_dir = [image for image in FOV_timepoint_images if timepoint in image][0]
                image = io.imread(image_dir)
                image = img_as_uint(skimage.transform.rotate(image, rotation))
                top_half, bottom_half = get_img_half(image,"top"), get_img_half(image,"bottom")
                bottom_half = img_as_uint(skimage.transform.rotate(bottom_half, 180))
                ## Do the top split of the FOV first
                top_threshold_line = trench_positions["{}_top".format(FOV)][0]
                bottom_threshold_line = trench_positions["{}_top".format(FOV)][1]
                midpoints = trench_positions["{}_top".format(FOV)][2]
                for y in range(len(midpoints)-1):
                    save_dir = output_directory + "trenches/" + FOV + "_top/" + channel + "/" + "trench_{}".format(str(y).zfill(2)) + "/T{}".format(timepoint) + ".png"
                    cropped_top = top_half[top_threshold_line:bottom_threshold_line,int(midpoints[y]):int(midpoints[y+1])]
                    imageio.imwrite(save_dir,cropped_top)
                ## Do the bottom split of the FOV next
                top_threshold_line = trench_positions["{}_bottom".format(FOV)][0]
                bottom_threshold_line = trench_positions["{}_bottom".format(FOV)][1]
                midpoints = trench_positions["{}_bottom".format(FOV)][2]
                for y in range(len(midpoints)-1):
                    save_dir = output_directory + "trenches/" + FOV + "_bottom/" + channel + "/" + "trench_{}".format(str(y).zfill(2)) + "/T{}".format(timepoint) + ".png"
                    cropped_top = bottom_half[top_threshold_line:bottom_threshold_line,int(midpoints[y]):int(midpoints[y+1])]
                    imageio.imwrite(save_dir,cropped_top)
    if double_mm == 0:
        for t in range(len(FOV_timepoint_images)):
            timepoint = re.findall("T(\d+)", FOV_timepoint_images[t])[0]
            image_dir = [image for image in FOV_timepoint_images if timepoint in image][0]
            image = io.imread(image_dir)
            image = img_as_uint(skimage.transform.rotate(image, rotation))
            top_threshold_line = trench_positions["{}".format(FOV)][0]
            bottom_threshold_line = trench_positions["{}".format(FOV)][1]
            midpoints = trench_positions["{}".format(FOV)][2]
            for y in range(len(midpoints)-1):
                save_dir = output_directory + "trenches/" + FOV + "/" + phase_channel + "/" + "trench_{}".format(str(y).zfill(2)) + "/T{}".format(timepoint) + ".png"
                cropped = image[top_threshold_line:bottom_threshold_line,int(midpoints[y]):int(midpoints[y+1])]
                imageio.imwrite(save_dir,cropped)
        for channel in other_channels:
            FOV_timepoint_images = get_image_list(directory,FOV, channel)
            for t in range(len(FOV_timepoint_images)):
                timepoint = re.findall("T(\d+)", FOV_timepoint_images[t])[0]
                image_dir = [image for image in FOV_timepoint_images if timepoint in image][0]
                image = io.imread(image_dir)
                image = img_as_uint(skimage.transform.rotate(image, rotation))
                top_threshold_line = trench_positions["{}".format(FOV)][0]
                bottom_threshold_line = trench_positions["{}".format(FOV)][1]
                midpoints = trench_positions["{}".format(FOV)][2]
                for y in range(len(midpoints)-1):
                    save_dir = output_directory + "trenches/" + FOV + "/" + channel + "/" + "trench_{}".format(str(y).zfill(2)) + "/T{}".format(timepoint) + ".png"
                    cropped = image[top_threshold_line:bottom_threshold_line,int(midpoints[y]):int(midpoints[y+1])]
                    imageio.imwrite(save_dir,cropped)
