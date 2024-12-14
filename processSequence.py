

from EightPoint import EightPoint
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from tqdm import tqdm
import pandas as pd

class Sequence:
    def __init__(self, directory):
        self.directory = directory
        self.images = self.__read_from_directory(directory)
        self.image_names = self.__get_image_names(directory)

    # Compute and plot the camera movement for the sequence of images
    # Method should be one of the following:
    # EightPoint.getRotationTranslationFromImagesOpenCV
    # EightPoint.getRotationTranslationFromImages
    # If iterative is True, the method calculates the camera position iteratively image by image
    # If iterative is False, the method calculates the camera relative to the first image
    def computeCameraMovement(self, K, method: Callable, iterative=True):
        # Initialize with the first camera position at the origin
        camera_positions = [np.zeros((3))]
        camera_rotations = [np.eye(3)]
        R_prev = np.eye(3)
        t_prev = np.zeros((1, 3))
        transformation = np.array([0, 0, 0, 0, 0, 0, 0])


        for i in tqdm(range(len(self.images) - 1)):

            if iterative:
                R, t, scaling_factor = method(self.images[i], self.images[i + 1], K)
                # Compute global camera position
                t_global = t_prev + R_prev @ t
                R_global = R @ R_prev
            else:
                R_global, t_global, scaling_factor = method(self.images[0], self.images[i + 1], K)
                print("t:", t_global)
                print("Scaling factor :", scaling_factor)
                # Assume camera is travelling in a straigh line, so vector should be (i+1) * t_global
                t_global = scaling_factor * t_global
                print("Scaled t:", t_global)
                
            camera_positions = np.append(camera_positions, [t_global.flatten()], axis=0)  # Update previous pose
            camera_rotations = np.append(camera_rotations, [R_global], axis=0) 
            transformation = np.vstack([transformation, np.hstack([i+1, t_global.flatten(), self.rotationMatrixToEulerAngles(R_global)])])
            print(np.hstack([i+1, t_global.flatten(), self.rotationMatrixToEulerAngles(R_global)]))
            print([t_global.flatten()])
            #print(camera_positions)
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
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], marker='o', linestyle='None', color='r')
        # plot the rotations as a unit vector rotation of (0,0,1)
        for i in range(len(camera_rotations)):
            z = np.array([0, 0, 1])
            z = camera_rotations[i]@z
            ax.quiver(camera_positions[i, 0], camera_positions[i, 1], camera_positions[i, 2], z[0], z[1], z[2], length=10, color='b')

        colors = plt.cm.viridis(np.linspace(0, 1, len(camera_positions)))

        for i, (x, y, z) in enumerate(camera_positions):
            ax.text(x, y, z, f'Image {i+1}', size=10, zorder=1, color=colors[i])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Positions of Camera for Each Image')

        # Set all axes to the same scale
        max_range = np.array([camera_positions[:, 0].max() - camera_positions[:, 0].min(),
                            camera_positions[:, 1].max() - camera_positions[:, 1].min(),
                            camera_positions[:, 2].max() - camera_positions[:, 2].min()]).max()

        mid_x = (camera_positions[:, 0].max() + camera_positions[:, 0].min()) * 0.5
        mid_y = (camera_positions[:, 1].max() + camera_positions[:, 1].min()) * 0.5
        mid_z = (camera_positions[:, 2].max() + camera_positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        plt.show()
        print(transformation)
        transformation_df = pd.DataFrame(transformation, columns=['frame', 'x', 'y', 'z', 'x_rot', 'y_rot', 'z_rot'])
        transformation_df.to_csv('transformation.csv', index=False)


    def rotationMatrixToEulerAngles(self,R) :
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])
        


    # This function returns a list of the names of all files in the Reference_Render_cubes folder
    def __get_image_names(self,directory):

        image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])

        # Return the list of image names
        return image_files[:100] + image_files[-1:]

    def __read_from_directory(self,directory):
        image_names = self.__get_image_names(directory)
        print(image_names)
        # Load the sequence of images
        images = [cv2.cvtColor( cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in image_names]
        return images



if __name__ == "__main__":

    sequence = Sequence('./Reference_Render_Translation')

    K = np.array([[2666.666666666, 0, 960],[0, 2666.666666666, 540],[
    0, 0, 1] ])
    eightP = EightPoint()

    sequence.computeCameraMovement(K, eightP.getRotationTranslationFromImagesOpenCV, iterative=False)
    

