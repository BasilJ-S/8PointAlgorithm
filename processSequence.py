

from EightPoint import EightPoint
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable

class Sequence:
    def __init__(self, directory):
        self.directory = directory
        self.images = self.__read_from_directory(directory)
        self.image_names = self.__get_image_names(directory)

    # Method should be one of the following:
    # EightPoint.getRotationTranslationFromImagesOpenCV
    # EightPoint.getRotationTranslationFromImages
    def computeCameraMovement(self, K, method: Callable):
        # Initialize with the first camera position at the origin
        camera_positions = [np.zeros((3))]  
        R_prev = np.eye(3)
        t_prev = np.zeros((1, 3))

        for i in range(len(self.images) - 1):

            R, t = method(self.images[i], self.images[i + 1], K)
            print(t_prev)
            print('-')
            print(t)
            print ('--')
            
            # Compute global camera position
            t_global = t_prev + R_prev @ t
            R_global = R @ R_prev
            print(t_global)
            print('---')

            camera_positions = np.append(camera_positions, [t_global.flatten()], axis=0)  # Update previous pose
            print(camera_positions)
            R_prev = R_global
            t_prev = t_global

        print(camera_positions)
        # Convert camera positions to numpy array
        camera_positions = np.array(camera_positions)

        # Check if the camera positions have the correct shape
        if camera_positions.shape[1] != 3:
            print("Warning: Camera positions have incorrect shape.")

        # Plot the 3D positions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', marker='o')

        colors = plt.cm.viridis(np.linspace(0, 1, len(camera_positions)))

        for i, (x, y, z) in enumerate(camera_positions):
            ax.text(x, y, z, f'Image {i+1}', size=10, zorder=1, color=colors[i])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Positions of Camera for Each Image')
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])
        ax.set_zlim([0, 50])

        plt.show()


    # This function returns a list of the names of all files in the Reference_Render_cubes folder
    def __get_image_names(self,directory):

        image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])

        # Return the list of image names
        return image_files

    def __read_from_directory(self,directory):
        image_names = self.__get_image_names(directory)
        print(image_names)
        # Load the sequence of images
        images = [cv2.cvtColor( cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in image_names]
        return images



if __name__ == "__main__":

    sequence = Sequence('./Reference_Render_Translation')
    print(sequence.image_names)

    # Retrieved from photo metadata for iPhone 13
    focal = 5.1
    pixPerMm = 2.835
    fm = focal * pixPerMm

    # Assume center of image is the principal point
    cx = sequence.images[0].shape[1] / 2
    cy = sequence.images[0].shape[0] / 2

    # Intrinsic matrix
    K = np.array([[fm, 0, cx],
                [0, fm, cy],
                [0, 0, 1]])

    #K = np.array([[2666.666666666, 0, 960],[0, 2666.666666666, 540],[
    #0, 0, 1] ])
    #eightP = EightPoint()
    eightP = EightPoint()

    sequence.computeCameraMovement(K, eightP.getRotationTranslationFromImagesRANSAC)
    

