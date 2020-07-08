import setup as config
from reading_data import Data
from skimage import exposure as ex
from scipy import ndimage
import numpy as np
import cv2


def scale_to_unit(src):
    max_value = np.max(src)
    min_value = np.min(src)
    output = (src.astype(np.float) - min_value) / max_value
    return output


def scale_to_255(src):
    max_value = np.max(src)
    min_value = np.min(src)
    output = 255 * (src.astype(np.float) - min_value) / max_value
    return output.astype(np.uint8)


def laplacian_filter(src, kernel_size=7):
    return cv2.Laplacian(src, cv2.CV_64F, ksize=kernel_size)


def sobel_x_filter(src, kernel_size=3):
    sobel = cv2.Sobel(src, cv2.CV_64F, 1, 0, kernel_size)
    return sobel


def sobel_y_filter(src, kernel_size=3):
    sobel = cv2.Sobel(src, cv2.CV_64F, 0, 1, kernel_size)
    return sobel


def he(img):
    output = ex.equalize_hist(img)
    return scale_to_unit(output)


def low_pass_filter(img):
    img_filtered = ndimage.gaussian_filter(img, 3)
    return img_filtered


def high_pass_filter(img):
    img_filtered = ndimage.gaussian_filter(img, 3)
    return img - img_filtered


def random_he(img):
    filtering = np.random.choice([True, False])
    if filtering:
        output = ex.equalize_hist(img)
    else:
        output = img
    return scale_to_unit(output)


if __name__ == '__main__':
    data = Data()
    id_class = 1
    while True:
        # img_org = data.read_random_images_by_label(label=classes[id_class], verbose=False)
        img_org = data.read_random_image_by_cat(cat="train", verbose=True)

        img_filtered = he(img_org)
        cv2.imshow("Filtered", img_filtered)
        cv2.imshow("Original Image", img_org)

        if cv2.waitKey(0) == 27:
            break
