import cv2
import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing
from skimage import measure


def compress_image(src: np.ndarray, scale_percent: int):
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    new_size = (width, height)

    return cv2.resize(src, new_size)


def extract_predetermined_items(img: np.ndarray):
    origin_image = np.copy(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur_gray = rgb2gray(gaussian(img, sigma=1.5, channel_axis=True))
    threshold_img = threshold_otsu(img_blur_gray)
    result_image = img_blur_gray <= threshold_img
    result_image = binary_closing(result_image, footprint=np.ones((20, 20)))

    mask, properties = get_smallest_component_mask(result_image)

    visMask = (mask * 255).astype("uint8")
    result_image = cv2.bitwise_and(origin_image, origin_image, mask=visMask)

    return result_image, draw_mask_contours(visMask, origin_image), properties


def extract_polygon_and_items(img: np.ndarray):
    height, width = img.shape

    img = canny(gaussian(img, 3), sigma=1.5, low_threshold=0.05)
    img = img[5:height - 5, 5:width - 5]
    img = binary_fill_holes(binary_closing(img, footprint=np.ones((10, 10))))

    polygon_img = img[0:int(height / 2), 0:width]
    items_img = img[int(height / 2):height, 0:width]

    return polygon_img, items_img


def get_smallest_component_mask(img: np.ndarray):
    labels = measure.label(img)
    properties = measure.regionprops(labels)

    item = np.array([element.area for element in properties]).argmin()

    return labels == item + 1, properties[item]


def draw_mask_contours(mask, image):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, [contours[0]], 0, (0, 255, 0), 3)

    return image
