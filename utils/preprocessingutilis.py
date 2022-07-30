import logging
from glob import glob
from PIL import Image, ImageEnhance
import matplotlib.pyplot as mpl
import cv2
import numpy as np
import os


def segment(im, thrd):
    # Input: image, wannted threshold value
    # Output: image after threshold
    width, height = im.size
    arr=np.asarray(im)
    arr_copy=arr.copy()
    for x in range(height):
        for y in range(width):
            if arr[x, y] <= thrd:
                arr_copy[x, y] = 0  # black
    return arr_copy

def otsu_thrd(im):
    ''' return the optimal threshold for a 256 gray level image im '''
    width, height = im.size
    hist = im.histogram(im)
    var_max = 0
    threshold=0
    for t in range(1,255): #t=0 and t=255 will yield 0 anyway
        back = sum(hist[0:t+1])
        fore = sum(hist[t+1:256])
        if back==0 or fore==0:
            continue
        mean_back = sum(hist[i]*i for i in range(t+1)) / back
        mean_fore = sum(hist[i]*i for i in range(t+1,len(hist))) / fore
        # Calculate Between Class Variance
        var_between = back * fore * (mean_back - mean_fore)**2
        # Check if new maximum found
        if (var_between > var_max):
            var_max = var_between
            threshold = t
    return threshold


def AVG(folder, avg_rate, dict_path):
    # Average function- averaging X normalized frames to one image
    # Input- folder and normal paths, number of frames in averaged image
    # Saving the averaging images in Avg folder
    image_files = sorted(glob('{}/*.tif'.format(dict_path['normal_path'])))
    logging.debug('%s files in folder %s',str(len(image_files)), dict_path['normal_path'])
    itr_max = len(image_files) - (len(image_files) % avg_rate)
    for i in range(0, itr_max, avg_rate):
        if i > itr_max:
            break
        avg_im = np.asarray(Image.open(image_files[i])).astype('float')
        for j in range(1, avg_rate):
            new_im = np.asarray(Image.open(image_files[i + j])).astype('float')
            avg_im = avg_im + new_im
        avg = (avg_im / avg_rate).astype('uint8')
        avg_path = os.path.join(folder, 'Avg' + str(avg_rate), 'Avg' + str(int(i / avg_rate)) + '.tif')
        if not os.path.isdir(os.path.join(folder, 'Avg' + str(avg_rate))):
            os.mkdir(os.path.join(folder, 'Avg' + str(avg_rate)))
        Image.fromarray(avg).save(avg_path)


def black_box_temp(folder):
    # Retriving black box body temperature
    # Input- folder name, Output- black box temp

    # extracting position of black box- pos[0]-col,pos[1]-row
    f = open(os.path.join(folder, 'Position.txt'))
    position = f.readline()
    position = position.split('\n')[0].split(';')
    x = int(position[1])
    y = int(position[0])
    # black box temp
    black_box = cv2.imread(os.path.join(folder, 'frames', 'frame_1.tiff'), 0)
    black_box = black_box[x][y]
    return black_box


def thresh_circle_norm(folder, bboxes, image_files, black_box,dict_path):
    # Thresholding, Circling and normlizing each frame
    # Input- folder path, Cascade model (bboxes), image frame list, black box body temp,paths list
    # Saving thresh_circle folder, saving normal folder

    # Face bounding circle coordinates
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height

    # Thresholding and bounding each frame
    for i, image_file in enumerate(image_files):
        image_new = Image.open(image_file).convert("L")
        # thresh.plt_histogram(image_new)
        t = otsu_thrd(image_new)
        im_thresh = segment(image_new, t)  # Image after threshold
        bounding_im = im_thresh[x:x2, y:y2]  # image after bounding circle
        image_name = 'frame_' + str(i) + '.tif'
        save_path = os.path.join(dict_path['thresh_circle_path'], image_name)
        if not os.path.isdir(os.path.join(folder, 'thresh_circle')):
            os.mkdir(dict_path['thresh_circle_path'])
        bounding_im = Image.fromarray(bounding_im)
        bounding_im.save(save_path)
        # Normlization
        normal_im = bounding_im.point(lambda p: (35 - (black_box - p) / 12.82))
        if not os.path.isdir(dict_path['normal_path']):
            os.mkdir(dict_path['normal_path'])
        normal_im.save(os.path.join(dict_path['normal_path'], image_name))


def path(folder,avg_rate,notDetectedPath):
    # Organising different paths in dictionary
    # Input- folder name, Output- paths dictionary
    path_dict = {'crop_path': os.path.join(folder, 'crop'),
                 'thresh_circle_path': os.path.join(folder, 'thresh_circle')
        , 'normal_path': os.path.join(folder, 'normal'),
                 'avg_path': os.path.join(folder, 'Avg' + str(avg_rate)),
                 'notDetectedPath': notDetectedPath}  # 'manual_path':os.path.join(path_, 'manual')}
    # Problem with manual path- needs manual assignment
    return path_dict


def cascade(image_files, dict_path):
    # Cascade function loads the cascade frontal face recognition (works good only on cropped images)
    # Input- list of all the frames of a patient, paths list. Output- loaded Cascade model
    classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    logging.debug('%s files in folder %s',str(len(image_files)), dict_path['crop_path'])
    bboxes = ()
    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file)
        # # rotating each image if needed
        # im = Image.fromarray(image)
        # w, h = np.size(im)
        # if h > w:
        #     im = im.transpose(Image.ROTATE_270)
        #     im.save(image_file)

        # Face detection in at least one frame using Cascade
        if len(bboxes) == 0:
            bboxes = classifier.detectMultiScale(image)
    return bboxes

def CONTRAST(image_files,output_path):

    for i, image_file in enumerate(image_files):
        im = Image.open(image_file).convert("L")
        enh = ImageEnhance.Contrast(im)
        # enh.enhance(15).show()
        image_name = 'contrast_' + str(i) + '.tif'
        enh.enhance(10).save(os.path.join(output_path,image_name))

def JET(image_files,path):
    for i, image_file in enumerate(image_files):
        img_src = Image.open(image_file).convert("L")
        cm_hot = mpl.cm.get_cmap('jet')
        img_src.thumbnail((512, 512))
        im = np.array(img_src)
        im = cm_hot(im)
        im = np.uint8(im * 255)
        im = Image.fromarray(im)
        rgb_im = im.convert('RGB')
        image_name = 'jet_' + str(i) + '.tif'
        rgb_im.save(os.path.join(path, image_name))

def ImgSpliting(image_files, path):
    for i, image_file in enumerate(image_files):
        im = Image.open(image_file).convert("L")
        width,height= im.size
        #box = (0, height - int(height / 2), height, width) #inferior face
        box = (0, 0, width, height - int(height / 2)) #superior face
        im=im.crop(box)
        image_name = 'sup_' + str(i) + '.tif'
        im.save(os.path.join(path, image_name))

