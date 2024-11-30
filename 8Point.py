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
    
    img3 = cv2.drawMatches(im1_gray, kp1, im2_gray, kp2, good_matches, None, **draw_params)
    inlier_pts1 = np.float32([kp1[m.queryIdx].pt for i, m in enumerate(good_matches) if matchesMask[i]])
    inlier_pts2 = np.float32([kp2[m.trainIdx].pt for i, m in enumerate(good_matches) if matchesMask[i]])
    
    return img3, inlier_pts1, inlier_pts2

# Function to find inlier matching points between two images
def find_inlier_matching_points(im1_gray, im2_gray):
    # Find the SIFT key points and descriptors in the two images
    kp1, des1 = find_sift_keypoints_and_descriptors(im1_gray)
    kp2, des2 = find_sift_keypoints_and_descriptors(im2_gray)

    # Match keypoints
    good = match_descriptors(des1, des2)

    # Find homography and draw matches
    img3, inlier_pts1, inlier_pts2 = find_homography_and_draw_matches(im1_gray, im2_gray, kp1, kp2, good)

    # Enable interactive mode
    plt.ion()

    # Show the matches image
    plt.figure()
    plt.imshow(img3, 'gray')
    plt.show()
    plt.ioff()

    return inlier_pts1, inlier_pts2

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

#-----------------Read Images-----------------#

im1 = cv2.imread('IMG_3125.jpeg')
im2 = cv2.imread('IMG_3126.jpeg')
im3 = cv2.imread('IMG_3127.jpeg')

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

pts1, pts2 = find_inlier_matching_points(im1_gray, im2_gray)
pts3,pts4 = find_inlier_matching_points(im2_gray,im3_gray)

F = compute_fundamental_matrix(pts1, pts2)

print("Fundamental Matrix F:")
print(F)



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


E = compute_essential_matrix(F, K)

print("Essential Matrix E:")
print(E)


#-----------------Recover Rotation and Translation-----------------#
# https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf

# There is ambiguity in finding which of R1 and R2 or t1 and t2 is the correct solution
# Need to add in checks to determine the correct solution

U, S, Vt = np.linalg.svd(E)

if np.linalg.det(U) < 0:
    U = -U
if np.linalg.det(Vt) < 0:
    Vt = -Vt

D = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
R1 = np.dot(np.dot(U, D), Vt)
R2 = np.dot(np.dot(U, D.T), Vt)
t1 = U[:, 2]
t2 = -U[:, 2]

print(f"T1: {t1}; T2: {t2}")

#Plot the points 0,0,0 and T1, T2 in 3D
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(0, 0, 0, color='r')
ax.text(0, 0, 0, "Camera 1")
ax.scatter(t1[0], t1[1], t1[2], color='b')
ax.text(t1[0], t1[1], t1[2], "Camera 2")

#fix axes in same scale
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plt.show()



