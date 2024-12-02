

from EightPoint import compute_rotation_translation, display_camera_movement
import os
import cv2
import numpy as np

# This function takes in a list of images and returns a list of camera translation matrices
def compute_camera_translations(images, K):
    # Initialize the list of camera translation matrices
    t_list = []

    # Iterate over the images
    for i in range(len(images) - 1):
        # Compute the camera rotation and translation between the current and next image
        R, t = compute_rotation_translation(images[i], images[i + 1], K)

        # Append the translation matrix to the list
        t_list.append(t)

    return t_list


# This function reads all images in the Reference_Render_cubes folder and returns a list of images
def read_images(image_names, directory):
    # Initialize the list of images
    images = []

    # Iterate over all images in the Reference_Render_cubes folder
    for i in range(0, 90):
        # Read the image
        img = cv2.imread(directory + '/' + image_names[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Append the image to the list
        images.append(img)

    # Return the list of images
    return images

# This function returns a list of the names of all files in the Reference_Render_cubes folder
def get_image_names(directory):
    # Initialize the list of image names
    image_names = []

    # Get the list of all files in the Reference_Render_cubes folder
    for filename in os.listdir(directory):
        # Append the file name to the list
        image_names.append(filename)

    # Return the list of image names
    return image_names

def read_from_directory(directory):
    image_names = get_image_names(directory)
    images = read_images(image_names, directory)
    return images



if __name__ == "__main__":

    # Read images from the Reference_Render_cubes folder
    images = read_from_directory('Reference_Render_cubes')
    images = images[0:7]

    # Retrieved from photo metadata for iPhone 13
    focal = 5.1
    pixPerMm = 2.835
    fm = focal * pixPerMm

    # Assume center of image is the principal point
    cx = images[0].shape[1] / 2
    cy = images[0].shape[0] / 2

    # Intrinsic matrix
    K = np.array([[fm, 0, cx],
                [0, fm, cy],
                [0, 0, 1]])

    #K = np.array([[2666.666666666, 0, 960],[0, 2666.666666666, 540],[
    #0, 0, 1] ])


    t_list = compute_camera_translations(images,K)
    display_camera_movement(t_list)
    print(t_list)

