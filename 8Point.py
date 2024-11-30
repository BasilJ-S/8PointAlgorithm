import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_MATCH_COUNT = 10


#-----------------Find SIFT Keypoints and Match with RANSAC-----------------#

# Detect SIFT keypoints in a greyscale image
def find_sift_keypoints_and_descriptors(image_gray):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors

# Find good matches between two sets of descriptors
def match_descriptors(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# Find homography between two images given the keypoints
def find_homography_and_draw_matches(im1_gray, im2_gray, kp1, kp2, good_matches):
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = im1_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(im2_gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    
    inlier_pts1 = np.float32([kp1[m.queryIdx].pt for i, m in enumerate(good_matches) if matchesMask[i]])
    inlier_pts2 = np.float32([kp2[m.trainIdx].pt for i, m in enumerate(good_matches) if matchesMask[i]])
    
    return inlier_pts1, inlier_pts2

# Function to find inlier matching points between two images
def find_inlier_matching_points(im1_gray, im2_gray):
    # Find the SIFT key points and descriptors in the two images
    kp1, des1 = find_sift_keypoints_and_descriptors(im1_gray)
    kp2, des2 = find_sift_keypoints_and_descriptors(im2_gray)

    # Match keypoints
    good = match_descriptors(des1, des2)

    # Find homography and draw matches
    inlier_pts1, inlier_pts2 = find_homography_and_draw_matches(im1_gray, im2_gray, kp1, kp2, good)

    return inlier_pts1, inlier_pts2

# Plot two images side by side with matches shown
def plot_two_images_with_matches(im1, im2, inlier_pts1, inlier_pts2):
    # Convert grayscale images to BGR so that we can draw colored lines
    im1_color = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    im2_color = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    img3 = np.hstack((im1_color, im2_color))
    for pt1, pt2 in zip(inlier_pts1, inlier_pts2):
        pt2 = (pt2[0] + im1.shape[1], pt2[1])
        cv2.line(img3, tuple(map(int, pt1)), tuple(map(int, pt2)), (0, 255, 0), 1)
        cv2.circle(img3, tuple(map(int, pt1)), 5, (0, 0, 255), -1)
        cv2.circle(img3, tuple(map(int, pt2)), 5, (0, 0, 255), -1)


    # Show the matches image
    plt.figure()
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

#-----------------Compute Fundamental Matrix-----------------#

# Compute fundamental matrix from inlier matches of two images
def compute_fundamental_matrix(pts1, pts2):

    # Normalize points
    pts1_mean = np.mean(pts1, axis=0)
    pts2_mean = np.mean(pts2, axis=0)
    pts1_std = np.std(pts1)
    pts2_std = np.std(pts2)

    T1 = np.array([[1/pts1_std, 0, -pts1_mean[0]/pts1_std],
                [0, 1/pts1_std, -pts1_mean[1]/pts1_std],
                [0, 0, 1]])

    T2 = np.array([[1/pts2_std, 0, -pts2_mean[0]/pts2_std],
                [0, 1/pts2_std, -pts2_mean[1]/pts2_std],
                [0, 0, 1]])

    pts1_normalized = np.dot(T1, np.vstack((pts1.T, np.ones((1, pts1.shape[0])))))
    pts2_normalized = np.dot(T2, np.vstack((pts2.T, np.ones((1, pts2.shape[0])))))

    # Construct matrix A for the normalized points
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1 = pts1_normalized[0, i], pts1_normalized[1, i]
        x2, y2 = pts2_normalized[0, i], pts2_normalized[1, i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    # Compute the fundamental matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0
    F_normalized = np.dot(U, np.dot(np.diag(S), Vt))

    # Denormalize the fundamental matrix
    F = np.dot(T2.T, np.dot(F_normalized, T1))
    return F

#-----------------Recover Essential Matrix-----------------#

# Recover essential matrix from fundamental matrix and camera intrinsic matrix
def compute_essential_matrix(F, K):
    E = np.matmul(np.matmul(K.T, F), K)
    return E

#-----------------Recover Rotation and Translation-----------------#
# https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf

# There is ambiguity in finding which of R1 and R2 or t1 and t2 is the correct solution
# Need to add in checks to determine the correct solution

# function to recover all valid R and t values from essential matrix E
def recover_rotation_translation(E):
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    D = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = np.dot(np.dot(U, D), Vt)
    R2 = np.dot(np.dot(U, D.T), Vt)
    t1 = U[:, 2]
    t2 = -U[:, 2]

    return R1, R2, t1, t2


# Disambiguate rotation and translation based on valid points and intrinsics
def disambiguate_rotation_translation(R1, R2, t1, t2, pts1, pts2, K):
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2_candidates = [
        np.hstack((R1, t1.reshape(-1, 1))),
        np.hstack((R1, t2.reshape(-1, 1))),
        np.hstack((R2, t1.reshape(-1, 1))),
        np.hstack((R2, t2.reshape(-1, 1))),
    ]

    max_valid_points_count = 0
    threshold = 10
    bestR = None
    bestT = None
    for i, P2 in enumerate(P2_candidates):
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        in_front_of_cam1 = points_3d[2, :] > 0
        in_front_of_cam2 = (P2 @ np.vstack((points_3d, np.ones((1, points_3d.shape[1])))))[2, :] > 0

        valid_points_count = np.sum(in_front_of_cam1 & in_front_of_cam2)

        if valid_points_count > threshold and valid_points_count > max_valid_points_count:
            print(f"Correct solution: R = {P2[:, :3]}, t = {P2[:, 3]}, valid points count = {valid_points_count}")
            max_valid_points_count = valid_points_count
            bestR = P2[:, :3]
            bestT = P2[:, 3]

    return bestR, bestT

# Function to compute camera rotation and translation from two images
def compute_rotation_translation(im1_gray, im2_gray, K):
    # Find inlier matching points
    pts1, pts2 = find_inlier_matching_points(im1_gray, im2_gray)

    # Plot the inlier matching points
    plot_two_images_with_matches(im1_gray, im2_gray, pts1, pts2)

    # Compute fundamental matrix
    F = compute_fundamental_matrix(pts1, pts2)

    # Compute essential matrix
    E = compute_essential_matrix(F, K)

    # Recover rotation and translation
    R1, R2, t1, t2 = recover_rotation_translation(E)

    # Disambiguate rotation and translation
    R, t = disambiguate_rotation_translation(R1, R2, t1, t2, pts1, pts2, K)

    return R,t

# Display the movement of the camera in 3D space from an array of translation matrices
def display_camera_movement(t_list):
    # Enable interactive mode
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the origin
    ax.scatter(0, 0, 0, color='r')
    ax.text(0, 0, 0, "Camera 1")

    # Initialize the current displacement
    current_displacement = np.zeros(3)

    # Plot the camera positions
    for i, t in enumerate(t_list):
        current_displacement += t
        ax.scatter(current_displacement[0], current_displacement[1], current_displacement[2], color='b')
        ax.text(current_displacement[0], current_displacement[1], current_displacement[2], f"Camera {i+2}")

    # Set the axes limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Show the plot
    plt.show()

#-----------------Main Code-----------------#

im1 = cv2.imread('IMG_3125.jpeg')
im2 = cv2.imread('IMG_3126.jpeg')
im3 = cv2.imread('IMG_3127.jpeg')

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

# Retrieved from photo metadata for iPhone 13
focal = 5.1
pixPerMm = 2.835
fm = focal * pixPerMm

# Assume center of image is the principal point
cx = im1_gray.shape[1] / 2
cy = im1_gray.shape[0] / 2

# Intrinsic matrix
K = np.array([[fm, 0, cx],
              [0, fm, cy],
              [0, 0, 1]])



Rt1, tt1 = compute_rotation_translation(im1_gray, im2_gray, K)
Rt2, tt2 = compute_rotation_translation(im2_gray, im3_gray, K)
Rt3, tt3 = compute_rotation_translation(im1_gray, im3_gray, K)

display_camera_movement([tt1, tt2])

"""
# Plot the camera positions in 3D space, for arbitrary 3D points
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(0, 0, 0, color='r')
ax.text(0, 0, 0, "Camera 1")
ax.scatter(tt1[0], tt1[1], tt1[2], color='b')
ax.text(tt1[0], tt1[1], tt1[2], "Camera 2")

ax.scatter(tt2[0] + tt1[0], tt2[1] + tt1[1], tt2[2] + tt1[2], color='g')
ax.text(tt2[0] + tt1[0], tt2[1] + tt1[1], tt2[2] + tt1[2], "Camera 3 from 2")  

ax.scatter(tt3[0], tt3[1], tt3[2], color='y')
ax.text(tt3[0], tt3[1], tt3[2], "Camera 3 from 1")


#fix axes in same scale
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

plt.show() 
"""




