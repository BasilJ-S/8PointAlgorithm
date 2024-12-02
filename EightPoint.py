import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_MATCH_COUNT = 8

class EightPoint:
    def __init__(self):
        pass
    #-----------------Find SIFT Keypoints and Match with RANSAC-----------------#

    # Detect SIFT keypoints in a greyscale image
    def __getSIFT(self,image_gray):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)
        return keypoints, descriptors

    # Find good matches between two sets of descriptors
    def __getMatches(self,des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches

    # Find homography between two images given the keypoints
    def __getInliers(self,kp1, kp2, good_matches):
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            matchesMask = None
        
        inlier_pts1 = np.float32([kp1[m.queryIdx].pt for i, m in enumerate(good_matches) if matchesMask[i]])
        inlier_pts2 = np.float32([kp2[m.trainIdx].pt for i, m in enumerate(good_matches) if matchesMask[i]])
        
        return inlier_pts1, inlier_pts2

    # Function to find inlier matching points between two images
    def getMatchingInliers(self,im1_gray, im2_gray):
        # Find the SIFT key points and descriptors in the two images
        kp1, des1 = self.__getSIFT(im1_gray)
        kp2, des2 = self.__getSIFT(im2_gray)

        # Match keypoints
        good = self.__getMatches(des1, des2)

        # Find homography and draw matches
        inlier_pts1, inlier_pts2 = self.__getInliers(kp1, kp2, good)

        return inlier_pts1, inlier_pts2

    # Plot two images side by side with matches shown
    def plotMatches(self,im1, im2, inlier_pts1, inlier_pts2):
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
    def getFundamental(self,pts1, pts2):

        # Normalize points
        pts1_mean = np.mean(pts1, axis=0)
        pts2_mean = np.mean(pts2, axis=0)
        pts1_std = np.std(pts1, axis=0)
        pts2_std = np.std(pts2, axis=0)

        T1 = np.array([[1/pts1_std[0], 0, -pts1_mean[0]/pts1_std[0]],
                    [0, 1/pts1_std[1], -pts1_mean[1]/pts1_std[1]],
                    [0, 0, 1]])

        T2 = np.array([[1/pts2_std[0], 0, -pts2_mean[0]/pts2_std[0]],
                    [0, 1/pts2_std[1], -pts2_mean[1]/pts2_std[1]],
                    [0, 0, 1]])

        pts1_normalized = T1 @ np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
        pts2_normalized = T2 @ np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))

        # Construct matrix A for the normalized points
        A = np.zeros((len(pts1), 9))
        for i in range(len(pts1)):
            x1, y1 = pts1_normalized[0, i], pts1_normalized[1, i]
            x2, y2 = pts2_normalized[0, i], pts2_normalized[1, i]
            A[i] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

        # Compute the fundamental matrix using SVD
        U, S, Vt = np.linalg.svd(A)
        F_normalized = Vt[-1].reshape(3, 3)

        # Enforce rank-2 constraint on F
        U, S, Vt = np.linalg.svd(F_normalized)
        S[2] = 0
        F_normalized = U @ np.diag(S) @ Vt

        # Denormalize the fundamental matrix
        F = T2.T @ F_normalized @ T1
        return F

    #-----------------Recover Essential Matrix-----------------#

    # Recover essential matrix from fundamental matrix and camera intrinsic matrix
    def getEssential(self,F, K):
        E = np.transpose(K) @ F @ K
        return E

    #-----------------Recover Rotation and Translation-----------------#
    # https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf

    # There is ambiguity in finding which of R1 and R2 or t1 and t2 is the correct solution
    # Need to add in checks to determine the correct solution

    # function to recover all valid R and t values from essential matrix E
    def __getCandidateTransform(self,E):
        U, S, Vt = np.linalg.svd(E)

        if np.linalg.det(U) < 0:
            U = -U
        if np.linalg.det(Vt) < 0:
            Vt = -Vt

        D = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        R1 = U@D@Vt
        R2 = U@D.T@Vt
        t1 = U[:, 2]
        t2 = -U[:, 2]

        return R1, R2, t1, t2


    # Disambiguate rotation and translation based on valid points and intrinsics
    def __disambiguateTransform(self,R1, R2, t1, t2, pts1, pts2, K):
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2_candidates = [
            np.hstack((R1, t1.reshape(-1, 1))),
            np.hstack((R1, t2.reshape(-1, 1))),
            np.hstack((R2, t1.reshape(-1, 1))),
            np.hstack((R2, t2.reshape(-1, 1))),
        ]

        best_valid_point_count = 0
        bestR = None
        bestT = None
        for i, P2 in enumerate(P2_candidates):
            
            # Normalize points using intrinsic matrix
            pts1_normalized = np.linalg.inv(K) @ np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T
            pts2_normalized = np.linalg.inv(K) @ np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T

            # Triangulate points
            points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_normalized[:2], pts2_normalized[:2]).T
            
            # If C1 < 0, the point is behind the first camera
            c1 = np.multiply(points_4d_hom[:,2],points_4d_hom[:,3])
            # Project points to second camera to get C2. If C2 < 0, the point is behind the second camera
            c2 = np.multiply((P2 @ points_4d_hom.T).T[:,2],points_4d_hom[:,3])
            c = np.column_stack((c1, c2))

            # Cound all points that are in front of both cameras
            validPoints = (c[:,0] > 0) & (c[:,1] > 0)
            numValidPoints = sum(validPoints)

            # Keep track of the best solution
            if numValidPoints >= best_valid_point_count:
                print(f"Correct solution: R = {P2[:, :3]}, t = {P2[:, 3]}, valid points count = {numValidPoints}")
                best_valid_point_count = numValidPoints
                bestR = P2[:, :3]
                bestT = P2[:, 3]

        # If no valid solution is found, return identiy matrices
        # This means if a change cannot be detected, we assume the camera is stationary
        if bestR is None or bestT is None:
            print("No valid solution found")
            bestR = np.eye(3,3)
            bestT = np.zeros(3)

        return bestR, bestT
    
    # Function to get the camera rotation and translation from the Essential matrix and the camera intrinsics
    def getRotationTranslationFromEK(self,E,K):
        R1, R2, t1, t2 = self.__getCandidateTransform(E)
        R,t = self.__disambiguateTransform(R1, R2, t1, t2, pts1, pts2, K)
        return R,t

    # Function to compute camera rotation and translation from two images
    def getRotationTranslationFromImages(self,im1_gray, im2_gray, K):
        # Find inlier matching points
        pts1, pts2 = self.getMatchingInliers(im1_gray, im2_gray)

        # Plot the inlier matching points
        #self.plotMatches(im1_gray, im2_gray, pts1, pts2)

        # Compute fundamental matrix
        F = self.getFundamental(pts1, pts2)

        # Compute essential matrix
        E = self.getEssential(F, K)

        # Recover rotation and translation
        R1, R2, t1, t2 = self.__getCandidateTransform(E)

        # Disambiguate rotation and translation
        R, t = self.__disambiguateTransform(R1, R2, t1, t2, pts1, pts2, K)

        return R,t
    
    # This function implements end to end camera pose estimation using OpenCV functions
    def getRotationTranslationFromImagesOpenCV(self, im1, im2, K):

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) > MIN_MATCH_COUNT:
            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

            E_CV, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R_CV, t_CV, mask = cv2.recoverPose(E_CV, pts1, pts2, K)
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            R_CV = np.eye(3)
            t_CV = np.zeros(3)
        
        return R_CV,t_CV.flatten()

# Display the movement of the camera in 3D space from an array of translation matrices
def plotCameraMovement(t_list):
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
        current_displacement += np.array(t, dtype=np.float64)
        ax.scatter(current_displacement[0], current_displacement[1], current_displacement[2], color='b')
        ax.text(current_displacement[0], current_displacement[1], current_displacement[2], f"Camera {i+2}")

    # Set the axes limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Show the plot
    plt.show()

#-----------------Main Code-----------------#
if __name__ == "__main__":
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
    
    eightP = EightPoint()
    
    pts1, pts2 = eightP.getMatchingInliers(im3_gray, im2_gray)
    
    # TEST - CHECK THAT FUNDAMENTAL MATRIX CALCULATION IS CORRECT
    F = eightP.getFundamental(pts1, pts2)

    F_CV, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT, 1.0, 0.999)
    print("Fundamental matrix from our implementation:")
    print(F)
    print("Fundamental matrix from OpenCV:")
    print(F_CV)
    print("Difference between the two matrices:")
    print(np.abs(F - F_CV))
    
    # TEST - compute the epipolar constraint error
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    error = np.abs(np.sum(pts2_hom * (F @ pts1_hom.T).T, axis=1))
    
    print("Epipolar constraint error (should be close to zero):")
    print(error)

    # TEST - Compute the essential matrix - INACCURATE
    E = eightP.getEssential(F, K)
    E_CV, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.9, threshold=1.0)

    print("Essential matrix from our implementation:")
    print(E)
    print("Essential matrix from OpenCV:")
    print(E_CV)
    print("Difference between the two matrices:")
    print(np.abs(E - E_CV))

    # Test - Recover rotation and translation

    R,t = eightP.getRotationTranslationFromEK(E,K)
    R,t = eightP.getRotationTranslationFromImages(im1_gray, im2_gray, K)
    _, R_CV, t_CV, mask = cv2.recoverPose(E, pts1, pts2, K)

    print("Rotation matrix from our implementation:")
    print(R)
    print("Rotation matrix from OpenCV:")
    print(R_CV)
    print("Difference between the two matrices:")
    print(np.abs(R - R_CV))

    print("Translation vector from our implementation:")
    print(t)
    print("Translation vector from OpenCV:")
    print(t_CV.T)
    print("Difference between the two vectors:")
    print(np.abs(t - t_CV.T))


    #Rt1, tt1 = compute_rotation_translation(im1_gray, im2_gray, K)
    #Rt2, tt2 = compute_rotation_translation(im2_gray, im3_gray, K)
    #Rt3, tt3 = compute_rotation_translation(im1_gray, im3_gray, K)

    #display_camera_movement([tt1, tt2])

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




