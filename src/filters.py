from setup import *
from src.reading_data import *


def laplacian_filter(src, kernel_size=3):
    src = cv2.GaussianBlur(src, (11, 11), 0)
    return cv2.Laplacian(src, cv2.CV_64F, ksize=kernel_size)


def sobel_x_filter(src, kernel_size):
    sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, kernel_size)
    return sobelx


if __name__ == '__main__':
    while True:
        image = read_random_images_by_label(label=2, verbose=False)
        cv2.imshow("Original Image", image)
        cv2.imshow("Laplacian", laplacian_filter(image))
        if cv2.waitKey(0) == 27:
            break
