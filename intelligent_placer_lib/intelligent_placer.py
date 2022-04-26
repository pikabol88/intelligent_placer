import imghdr
from os import listdir, path

from intelligent_placer_lib.basic_item import *
from intelligent_placer_lib.image_processing import compress_image, extract_polygon_and_items

scale_ratio = 20


def process_predetermined_items(path_to_folder: str, path_for_results: str = "") -> list:
    processed_items = []

    for image_path in listdir(path_to_folder):
        image_full_path = path.join(path_to_folder, image_path)

        if imghdr.what(image_full_path) == 'jpeg':
            img = cv2.imread(image_full_path)
            img = compress_image(img, scale_ratio)

            result = Item(img, image_path)
            processed_items.append([(result.mask * 255).astype("uint8"), result.contour_image])

            if len(path_for_results) > 0:
                image_name = f"{path_for_results}/{image_path.split('/')[-1]}"
                cv2.imwrite(image_name, result.contour_image)

    return processed_items


def check_image(path_to_img: str, path_for_results: str = "") -> bool:
    if imghdr.what(path_to_img) != 'jpeg':
        return False

    img = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_RGB2GRAY)
    img = compress_image(img, scale_percent=scale_ratio)

    result, result_img = _process_image(img)

    if result and len(path_for_results) > 0:
        new_path = path.join(path_for_results, f"{str(result)}_{path.basename(path_to_img)}")
        cv2.imwrite(new_path, (result_img * 255).astype("uint8"))

    return result


def _process_image(img: np.ndarray) -> [bool, np.ndarray]:
    result = np.ndarray

    polygon_img, items_img = extract_polygon_and_items(img)
    polygon = Polygon(polygon_img, "polygon")
    items_set = ItemsSet(items_img)

    if sum(element.properties.area for element in items_set.items) > polygon.properties[0].area:
        return False, None

    for el in items_set.items:
        result = _try_place_item(polygon.mask, el.mask, el.properties)
        if result is None:
            return False, result

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


