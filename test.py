import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def extract_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def draw_matches(img1, kp1, img2, kp2, matches):
    match_color = (0, 0, 255)  # Red points
    
    # Draw the matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  matchColor=match_color,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Convert the image to RGB
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    # Plot the matches
    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches)
    plt.title('Feature Matching')
    plt.show()

def estimate_camera_poses(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def main():
    # Path to the images
    image_folder = 'Reference_Render_cubes'
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])

    # Load the sequence of images
    images = [cv2.imread(img) for img in image_files]

    # Retrieved from photo metadata for iPhone 13
    focal = 5.1
    pixPerMm = 2.835
    fm = focal * pixPerMm

    # Assume center of image is the principal point
    cx = images[0].shape[1] / 2
    cy = images[0].shape[0] / 2
    
    # Intrinsic matrix (Iphone 13)
    K = np.array([[fm, 0, cx],
                [0, fm, cy],
                [0, 0, 1]])
    """
    # Intrinsic matrix (Joey's estimation)
    K = np.array([[2666.666666666, 0, 960],
                [0, 2666.666666666, 540],
                [0, 0, 1]])
    """

    # Initialize with the first camera position at the origin
    camera_positions = [np.zeros((3,))]  
    R_prev = np.eye(3)
    t_prev = np.zeros((3, 1))

    for i in range(len(images) - 1):
        # Extract features from the current and next image
        kp1, des1 = extract_features(images[i])
        kp2, des2 = extract_features(images[i + 1])

        # Match features
        matches = match_features(des1, des2)

        # Proceed only if there are valid matches
        if len(matches) > 0:
            # Get matched points
            pts1, pts2 = get_matched_points(kp1, kp2, matches)

            # Estimate camera pose
            R, t = estimate_camera_poses(pts1, pts2, K)

            # Compute global camera position
            t_global = t_prev + R_prev @ t
            R_global = R @ R_prev

            camera_positions.append(t_global.flatten())

            # Update previous pose
            R_prev = R_global
            t_prev = t_global
        else:
            print(f"No valid matches found for image pair {i} and {i + 1}. Skipping pose estimation.")

    # Convert camera positions to numpy array
    camera_positions = np.array(camera_positions)

    # Check if the camera positions have the correct shape
    if camera_positions.shape[1] != 3:
        print("Warning: Camera positions have incorrect shape.")

    # Plot the 3D positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', marker='o')

    for i, (x, y, z) in enumerate(camera_positions):
        ax.text(x, y, z, f'Image {i+1}', size=10, zorder=1, color='k')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Positions of Camera for Each Image')

    plt.show()
  
def plot_matches():
    img1 = cv2.imread('Reference_Render_cubes\cubes0001.png')
    img2 = cv2.imread('Reference_Render_cubes\cubes0090.png')

    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)

    matches = match_features(des1, des2)
    print(len(matches))

    # Draw the matches
    draw_matches(img1, kp1, img2, kp2, matches)

if __name__ == "__main__":
    #main() # Uncomment to plot camera positions
    plot_matches()
