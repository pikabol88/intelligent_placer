import abc
import cv2
import numpy as np

from skimage.measure import regionprops, label as sk_measure_label
from image_processing import extract_predetermined_items


class BasicItem:
    def __init__(self, item_image: np.ndarray, item_name: str, properties=None, mask=None):
        self._name = item_name
        self._original_image = item_image

        self._properties = properties
        self._mask = mask

    @abc.abstractmethod
    def _set_characteristics(self):
        return

    @property
    def image(self):
        return self._original_image

    @property
    def properties(self):
        return self._properties

    @properties.setter
    def properties(self, properties):
        self._properties = properties

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask


class Item(BasicItem):
    def __init__(self, item_image: np.ndarray, item_name: str):
        super().__init__(item_image, item_name)
        self.name = item_name
        self._path = item_name  # remove

        self._processed_image = None
        self._contour_image = None

        self._set_characteristics()

    def _set_characteristics(self):
        self._processed_image, self._contour_image, self.properties = extract_predetermined_items(self._original_image)
        self.mask = cv2.normalize(np.array(self.properties.image, dtype=np.int32), None, 0, 255,
                                  cv2.NORM_MINMAX).astype('uint8')
        cv2.imwrite(f"{self._path}", self._contour_image)

    @property
    def contour_image(self):
        return self._contour_image

    @property
    def processed_image(self):
        return self._processed_image


class Polygon(BasicItem):
    def __init__(self, item_image: np.ndarray, item_name: str):
        super().__init__(item_image, item_name)

        self._set_characteristics()

    def _set_characteristics(self):
        labels = sk_measure_label(self.image)
        self._properties = regionprops(labels)
        if len(self._properties) != 1:
            pass
        self.mask = self._properties[0].image


class ItemsSet:
    def __init__(self, items_image: np.ndarray):
        self._items_list = []
        self._properties = regionprops(sk_measure_label(items_image))

        for i in range(len(self._properties)):
            mask = cv2.normalize(np.array(self._properties[i].image, dtype=np.int32), None, 0, 255,
                                 cv2.NORM_MINMAX).astype('uint8')
            self._items_list.append(
                BasicItem(self._properties[i].image, f"{i}", self._properties[i], mask))

    @property
    def items(self):
        return self._items_list
