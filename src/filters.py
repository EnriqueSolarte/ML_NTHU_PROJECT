from setup import *
from src.reading_data import *
from skimage import exposure as ex


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

        # img_blurred = cv2.GaussianBlur(img_org, (3, 3), cv2.CV_64F, 1)
        # img_blurred = img_org
        # cv2.imshow("Filtered Image", he(img_blurred))
        # img_lap = scale_to_unit(scale_to_unit(laplacian_filter(img_blurred)) + scale_to_unit(img_blurred))

        cv2.imshow("Sharp", he(img_org))
        cv2.imshow("Original Image", img_org)

        if cv2.waitKey(0) == 27:
            break
