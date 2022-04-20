import sys
from os import listdir
from sys import path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.morphology import binary_closing, binary_opening
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu

from skimage.filters import gaussian

folder_path = "C:/Users/nena2/Documents/GitHub/intelligent_placer/items"

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def process_images():
    for image_path in listdir(folder_path):
        image = cv2.imread(f"{folder_path}/{image_path}")

        image = compress_image(image, 30)

        result, props = extract_object(image)
        im = np.array(props.image, dtype=np.int32)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        cv2.imshow("l", im)
        cv2.waitKey(0)
        cv2.imwrite(f"{image_path}", result)
        cv2.imshow('mask', img3)
        k = cv2.waitKey(0)


def compress_image(src: np.ndarray, scale_percent: int):
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    new_size = (width, height)
    return cv2.resize(src, new_size)


def extract_object(img: np.ndarray):
    origin_image = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur_gray = rgb2gray(gaussian(img, sigma=1.5, channel_axis=True))
    threshold_img = threshold_otsu(img_blur_gray)
    result_image = img_blur_gray <= threshold_img
    result_image = binary_closing(result_image, footprint=np.ones((20, 20)))

    mask, props = get_smallest_component_mask(result_image)
    visMask = (mask * 255).astype("uint8")
    result_image = cv2.bitwise_and(origin_image, origin_image, mask=visMask)
    draw_mask_contours(visMask, origin_image)
    return result_image, props


def get_smallest_component_mask(img: np.ndarray):
    labels = measure.label(img)
    properties = measure.regionprops(labels)
    item = np.array([element.area for element in properties]).argmin()
    return labels == item + 1, properties[item]


def draw_mask_contours(mask, image):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)
    # Откройте контурное изображение
    cv2.imshow('mask', image)
    k = cv2.waitKey(0)





if __name__ == '__main__':
    process_images()
