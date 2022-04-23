import os

import matplotlib.pyplot as plt
from os import listdir

from intelligent_placer_lib.basic_item import *
from intelligent_placer_lib.image_processing import compress_image, extract_polygon_and_items

scale_ratio = 20


def process_predetermined_items(path_to_folder: str):
    items = []

    for image_path in listdir(path_to_folder):
        img = cv2.imread(f"{path_to_folder}/{image_path}")
        img = compress_image(img, scale_ratio)
        items.append(Item(img, image_path))


def check_image(path: str, path_for_results:str) -> bool:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
    img = compress_image(img, scale_percent=scale_ratio)

    result, result_img = _process_image(img)

    if result:
        res = (result_img * 255).astype("uint8")
        img_name = path.split("\\")[-1]
        new_path = f"{path_for_results}/{str(result)}{img_name}"
        cv2.imwrite(new_path, res)

    return result


def _process_image(img: np.ndarray) -> [bool, np.ndarray]:
    result = np.ndarray

    polygon_img, items_img = extract_polygon_and_items(img)
    polygon = Polygon(polygon_img, "polygon")
    items_set = ItemsSet(items_img)

    total_area = np.array([element.properties.area for element in items_set.items])
    if total_area.sum() > polygon.properties[0].area:
        return False, None

    for el in items_set.items:
        result = _try_place_item(polygon.mask, el.mask, el.properties)
        if result is None:
            return False, result

   # plt.imshow(result, cmap='gray')
   # plt.show()
    return True, result


def _try_place_item(place_for_items, item_mask, item_properties, step=5) -> np.ndarray:
    place_y, place_x = place_for_items.shape
    item_y, item_x = item_mask.shape

    for pos_y in range(0, place_y - item_y, step):
        for pos_x in range(0, place_x - item_x, step):

            item_contour_box = place_for_items[pos_y:pos_y + item_y, pos_x:pos_x + item_x].astype(int)
            bitwiseAnd = cv2.bitwise_and(item_contour_box.astype("uint8"), item_mask.astype("uint8"))

            if np.sum(bitwiseAnd) == item_properties.area:
                place_for_items[pos_y:pos_y + item_y, pos_x:pos_x + item_x] = cv2.bitwise_not(item_mask)
                return place_for_items

    return None


if __name__ == '__main__':
    path = "C:/Users/nena2/Documents/GitHub/intelligent_placer/test_dataset"
    result_path = "C:/Users/nena2/Documents/GitHub/intelligent_placer/results"
    for filename in os.listdir(path):
        check_image(str(os.path.join(path, filename)), result_path)
