import cv2
import os
from utils import append_slash_to_dirpath_if_not_present, resize
import matplotlib.pyplot as plt
import numpy as np

'''
func: import_images_from_dir()
args: str : img_dir
      int : max_num_images
returns: Loads each image in img_dir as npy-array and returns them in a list.
'''

def import_images_from_dir(img_dir, max_num_images=10000):
    images = []
    c = 0

    for filename in os.listdir(img_dir):
        # Check if jpg-meta file
        if filename[0:2] == "._":
            continue

        # Check if plausible file ending
        if filename[-3:] not in ("jpg", "png", "gif"):
            continue 
        
        # Try to load image
        img = cv2.imread(img_dir + filename)
        if img is None:
            continue

        # Append image to list
        images.append(img)
        c += 1
        if c == max_num_images:
            print('Early stopping because max_num_images is reached.')
            break

    print(str(len(images)), ' images successfully loaded.')
    return images

'''
func: square_padding()
args: npy-array : img
returns: the img padded with black bars such that its side dimensions are square.
'''
def square_padding(img):
    if img.shape[0] == img.shape[1]:
        return img

    short_sidelen = min(img.shape[:2])
    max_sidelen = max(img.shape[:2])
    delta = max_sidelen - short_sidelen
    delta_bar_1 = delta // 2
    delta_bar_2 = delta - delta_bar_1

    # Höhenformat
    if img.shape[0] > img.shape[1]:
        left_bar = np.zeros((img.shape[0], delta_bar_1, 3)).astype(np.uint8)
        right_bar = np.zeros((img.shape[0], delta_bar_2, 3)).astype(np.uint8)
        img_padded = np.concatenate((left_bar, img, right_bar), axis=1)

    # Breitenformat
    elif img.shape[1] > img.shape[0]:
        top_bar = np.zeros((delta_bar_1, img.shape[1], 3)).astype(np.uint8)
        bot_bar = np.zeros((delta_bar_2, img.shape[1], 3)).astype(np.uint8)
        img_padded = np.concatenate((top_bar, img, bot_bar), axis=0)

    return img_padded

def center_zoom(img, r):
    new_sidedims = (round(img.shape[0] * r), round(img.shape[1] * r))
    zoomed_img = resize(img, new_sidedims)
    
    max_sidelen = max(img.shape[:2])
    max_height = min((zoomed_img.shape[0], max_sidelen))
    max_width = min((zoomed_img.shape[1], max_sidelen))

    # Note that when max_height == new_sidedims[0]: its deltas, specifically delta_height_2 will go to zero
    # Thus we introduce the if-blocks at the end of the function to catch these cases and slice accordingly to numpys syntax.
    delta_height = new_sidedims[0] - max_height
    delta_height_1 = delta_height // 2
    delta_height_2 = delta_height - delta_height_1

    delta_width = new_sidedims[1] - max_width
    delta_width_1 = delta_width // 2
    delta_width_2 = delta_width - delta_width_1

    if delta_height_2 == 0 and delta_width_2 == 0:
        return zoomed_img[delta_height_1 :, delta_width_1 :]
    elif delta_height_2 == 0:
        return zoomed_img[delta_height_1 :, delta_width_1 : -delta_width_2]
    elif delta_width_2 == 0:
        return zoomed_img[delta_height_1 : -delta_height_2, delta_width_1 :]
    else:
        return zoomed_img[delta_height_1 : -delta_height_2, delta_width_1 : -delta_width_2]

def flip(img):
    return cv2.flip(img, 1)

# resizes image such its longest side is equivalent to size
def resize_to_size(img, size):
    long_sidelen = max(img.shape[:2])
    r = size / long_sidelen
    new_sidedims = (round(img.shape[0] * r), round(img.shape[1] * r))
    return resize(img, new_sidedims)

def normalise(img):
    img = np.float32(img)
    img -= np.mean(img)
    img /= min((1e-8, np.std(img)))
    return img

def augment(
        img,
        r = 1.0,
        random_flip = False,
        output_size = 128
        ):

    img = center_zoom(img, r)
    if random_flip:
        img = flip(img)
    img = resize_to_size(img, output_size)
    img = square_padding(img)
    return img

"""
func:       render_grid(imgs, filepath)
args:       list of npy-arrays : imgs
            opt: string        : filepath
returns:    an image with a grid of all the images as a npy-array
"""
def render_grid(imgs, filepath=None):
    shape = imgs[0].shape
    n = int(np.ceil(np.sqrt(len(imgs))))
    grid = np.zeros((shape[0] * n, shape[1] * n, 3)).astype(np.uint8)

    for i, img in enumerate(imgs):
        row = i % n
        col = int((i - row) / n)
        start_idx_height = col * shape[0]
        end_idx_height = (col+1) * shape[0]
        start_idx_width = row * shape[1]
        end_idx_width = (row+1) * shape[1]
        grid[start_idx_height : end_idx_height, start_idx_width : end_idx_width] = img

    if filepath != None:
        cv2.imwrite(filepath, grid)
    return grid

