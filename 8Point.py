import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_MATCH_COUNT = 10
#-----------------Read Images-----------------#

im1 = cv2.imread('IMG_3125.jpeg')
im2 = cv2.imread('IMG_3126.jpeg')
im3 = cv2.imread('IMG_3127.jpeg')

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

#-----------------Find SIFT Keypoints and Match with RANSAC-----------------#

# Find the SIFT key points and descriptors in the two images
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1_gray, None)
kp2, des2 = sift.detectAndCompute(im2_gray, None)
kp3, des3 = sift.detectAndCompute(im3_gray, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
 
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
 
# cv2.drawMatchesKnn expects list of lists as matches.
#matchImage = cv2.drawMatchesKnn(im1_gray,kp1,im2_gray,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(matchImage),plt.show()


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = im1_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(im2_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(im1_gray,kp1,im2_gray,kp2,good,None,**draw_params)
# Enable interactive mode
plt.ion()

# Show the matches image
plt.figure()
plt.imshow(img3, 'gray')
plt.show()
plt.ioff()
#-----------------Compute Fundamental Matrix-----------------#

# Extract points from the good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

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
A = np.zeros((len(good), 9))
for i in range(len(good)):
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

print("Fundamental Matrix F:")
print(F)


#-----------------Recover Essential Matrix-----------------#

# Retrieved from photo metadata for iPhone 13
focal = 5.1
pixPerMm = 2.835
fm = focal * pixPerMm

# Assume center of image is the principal point
cx = im1_gray.shape[1] / 2
cy = im1_gray.shape[0] / 2

K = np.array([[fm, 0, cx],
              [0, fm, cy],
              [0, 0, 1]])

inverseK = np.linalg.inv(K)

E = np.matmul(np.matmul(K.T, F), K)
print("Essential Matrix E:")
print(E)


#-----------------Recover Rotation and Translation-----------------#
# https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf

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



