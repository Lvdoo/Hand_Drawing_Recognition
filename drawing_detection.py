import cv2 as cv
import numpy as np


# In openCV, lignes = y and columns = x
def locate_drawing(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    height, width = gray_image.shape[:2]
    inverted_image = invert_color(gray_image)
    clean_image = clean_noise(inverted_image)
    drawing_pixels= locate_number(clean_image)
    if len(drawing_pixels) == 0 :
        return None, None # 2 None because the function returns 2 elements

    x_min, y_min, x_max, y_max = get_bounding_box(drawing_pixels)
    x_min, y_min, x_max, y_max = add_margin(width, height, x_min, y_min, x_max, y_max)
    cropped_image = crop(clean_image, x_min, y_min, x_max, y_max)    
    # cv.imwrite('cleaned_image.png', cropped_image)
    return cropped_image, (x_min, y_min, x_max, y_max)

def invert_color(image) :
    new_image = cv.adaptiveThreshold(src = image, maxValue = 255, adaptiveMethod = cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv.THRESH_BINARY_INV, blockSize = 51, C = 5)
    # cv.imwrite('inverted_image.png', new_image)

    return new_image

def clean_noise(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image

    largest_contour = max(contours, key=cv.contourArea)
    mask = np.zeros_like(image) # Create a black image
    cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    clean_image = cv.bitwise_and(image, mask) # COmpare every pixels of the two images: pixel_image 1 = [255,0,0], pixel_image 2 [255, 255, 0], result [255, 0, 0]

    return clean_image

def locate_number(image) :
    drawing_pixels = cv.findNonZero(image) # Find all the pixels diffrent from 0
    if drawing_pixels is None :
        list_tuples = []
        return list_tuples
    list_tuples = list(map(tuple, np.reshape(drawing_pixels,(-1,2))))
    
    return list_tuples

def get_bounding_box(drawing_pixels) :
    x_list = [coord[0] for coord in drawing_pixels]
    y_list = [coord[1] for coord in drawing_pixels]

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    return x_min, y_min, x_max, y_max

def add_margin(width, height, x_min, y_min, x_max, y_max) :
    margin = 15
    x_min_margin = max(x_min - margin, 0)
    y_min_margin = max(y_min - margin, 0)
    x_max_margin = min(x_max + margin, width)
    y_max_margin = min(y_max + margin, height)

    return x_min_margin, y_min_margin, x_max_margin, y_max_margin

def crop(image, x_min_margin, y_min_margin, x_max_margin, y_max_margin) :   
    cropped_image = image[y_min_margin : y_max_margin, x_min_margin : x_max_margin]
    return cropped_image