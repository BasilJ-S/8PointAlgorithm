import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import apply_matrix, calculate_dist

MIN_MATCH_COUNT = 8

class EightPoint:
    def __init__(self):
        pass
    #-----------------Find SIFT Keypoints and Match with RANSAC-----------------#

    # Function to find the matching points between two images using OpenCV functions
    def getMatchingPointsOpenCV(self,im1, im2):
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2

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

    def apply_matrix(pts, F):
        ''' 
        apply matrix F to the input points, first converting to homogenous
        Params:
        pts: (Nxm) matrix of points
        F: (m+1 x m+1) matrix (could be fundmental)
        '''  
        # Convert to homgenous coords  
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1)))) # (N x m)

        # Transform by matrix F
        transformed_pts = (F @ pts_hom.T).T  # (N x m)

        # Divide by homogenous scaling
        return transformed_pts[:, 0:pts.shape[1]] / transformed_pts[:, pts.shape[1]][:, None]

    # Get normalizing matrix for a set of points
    def __getNormalizingMatrix(self,pts):
        # Normalize points
        ptsMean = np.mean(pts, axis=0)
        ptsCentered = pts - ptsMean
        ptsStd = np.sqrt(np.sum(np.sum(ptsCentered ** 2, axis=1))/(2*len(pts)))
        M = np.array([[1/ptsStd, 0, 0],
                    [0, 1/ptsStd, 0],
                    [0, 0, 1]]) @ np.array([[1, 0, -ptsMean[0]],
                                            [0, 1, -ptsMean[1]],
                                            [0, 0, 1]])
        return M

    # Get the fundamental matrix using a least squares approach. If eightPoints is set to True, only 8 points are used
    def getFundementalLS(self,pts1, pts2, eightPoints = False):
        if eightPoints and (len(pts1) != 8 or len(pts2) != 8):
            print("Error: Only 8 points are allowed for the 8-point algorithm")
            return

        M1 = self.__getNormalizingMatrix(pts1)
        M2 = self.__getNormalizingMatrix(pts2)

        pts1Normalized = M1 @ np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
        pts2Normalized = M2 @ np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))

        # Construct matrix A for the normalized points
        A = np.zeros((max(len(pts1), 9), 9))
        for i in range(len(pts1)):
            x1, y1 = pts1Normalized[0, i], pts1Normalized[1, i]
            x2, y2 = pts2Normalized[0, i], pts2Normalized[1, i]
            A[i] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
        if eightPoints:
            A[8] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Compute the fundamental matrix using SVD
        U, S, Vt = np.linalg.svd(A)
        Fnormalized = Vt[-1].reshape(3, 3)

        # Enforce rank-2 constraint on F
        U, S, Vt = np.linalg.svd(Fnormalized)
        S[2] = 0
        Fnormalized = U @ np.diag(S) @ Vt

        # Denormalize the fundamental matrix
        F = M2.T @ Fnormalized @ M1
        # Normalize the fundamental matrix
        F = F/F[2,2]
        return F
    
    # Use RANSAC to find the fundamental matrix between two images, taking in all matches as input (NOT INLIER MATCHES)
    def getFundamentalRANSAC(self,pts1,pts2):
        bestF = np.eye(3)
        bestInliers = 0
        concensusSetMinSize = 9
        concensusMaxError = 1
        inlierMaxError = 0.1

        if not (len(pts1) == len(pts2) and len(pts1) >=8):
            raise Exception("Not enough points for ransac")

        for i in range(100):   
            # Randomly select 8 points

            idx = np.random.choice(len(pts1), 8, replace=False)
            pts1_sample = pts1[idx]
            pts2_sample = pts2[idx]

            # Compute the fundamental matrix using least squares
            F_consensus = self.getFundementalLS(pts1_sample, pts2_sample, eightPoints=True)

            # Compute the epipolar constraint error
            pts1_transformed = apply_matrix(pts1, F_consensus)
            error = calculate_dist(pts1_transformed, pts2)

            consensus_set_indices = error < concensusMaxError
            # Count the number of inliers
            inliers = np.sum(error < concensusMaxError)
            if inliers > concensusSetMinSize:
                # Recalculate F from all points in the consensus set
                F_consensus = self.getFundementalLS(pts1[consensus_set_indices, :], pts2[consensus_set_indices, :])
                inliers = np.sum(error < inlierMaxError)
                if inliers > bestInliers:
                    bestInliers = inliers
                    print(f"Best number of inliers: {bestInliers}, outliers: {len(pts1) - bestInliers}")
                    bestF = F_consensus
        return bestF
    
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

        """ # Plot t1, t2, R1, R2. T1, t2 are points, R1 and R2 should be displayed as unit vectors of just the Z direction
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot t1 and t2
        ax.scatter(t1[0], t1[1], t1[2], color='r', label='t1')
        ax.scatter(t2[0], t2[1], t2[2], color='b', label='t2')

        # Plot R1 and R2 as unit våectors in the Z direction at t1 and t2
        ax.quiver(t1[0], t1[1], t1[2], R1[0, 2], R1[1, 2], R1[2, 2], color='g', label='R1 at t1')
        ax.quiver(t1[0], t1[1], t1[2], R2[0, 2], R2[1, 2], R2[2, 2], color='y', label='R2 at t1')
        ax.quiver(t2[0], t2[1], t2[2], R1[0, 2], R1[1, 2], R1[2, 2], color='g', linestyle='dashed', label='R1 at t2')
        ax.quiver(t2[0], t2[1], t2[2], R2[0, 2], R2[1, 2], R2[2, 2], color='y', linestyle='dashed', label='R2 at t2')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        ax.scatter(0, 0, 0, color='k', s=100, label='Origin')
        # Add quiver in the direction of the Z axis
        ax.quiver(0, 0, 0, 0, 0, 1, color='k', label='Z axis')
        ax.legend(fontsize=14)
        plt.show() """


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
                #print(f"Correct solution: R = {P2[:, :3]}, t = {P2[:, 3]}, valid points count = {numValidPoints}")
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

    def getRotationTranslationFromImagesRANSAC(self,im1_gray, im2_gray, K):
        # Find inlier matching points
        pts1, pts2 = self.getMatchingPointsOpenCV(im1_gray, im2_gray)

        # Plot the inlier matching points
        #self.plotMatches(im1_gray, im2_gray, pts1, pts2)

        # Compute fundamental matrix
        F = self.getFundamentalRANSAC(pts1, pts2)

        # Compute essential matrix
        E = self.getEssential(F, K)

        # Recover rotation and translation
        R1, R2, t1, t2 = self.__getCandidateTransform(E)

        # Disambiguate rotation and translation
        R, t = self.__disambiguateTransform(R1, R2, t1, t2, pts1, pts2, K)

        scaling_factor = np.linalg.norm(np.median(pts1-pts2, axis=0))
        return R,t,scaling_factor
    
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
            # Apply the mask to the points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            # R1, R2, t1, t2 = self.__getCandidateTransform(E_CV)
            # R_CV, t_CV = self.__disambiguateTransform(R1, R2, t1, t2, pts1, pts2, K)
            _, R_CV, t_CV, mask = cv2.recoverPose(E_CV, pts1, pts2, K)
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            R_CV = np.eye(3)
            t_CV = np.zeros(3)
        scaling_factor = np.linalg.norm(np.median(pts1-pts2, axis=0))
        return R_CV,t_CV.flatten(), scaling_factor


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
    
    pts1, pts2 = eightP.getMatchingPointsOpenCV(im3_gray, im2_gray)
    # -----------FUNDAMENTAL MATRIX ESTIMATION----------------
    # TEST - CHECK THAT FUNDAMENTAL MATRIX CALCULATION IS CORRECT
    F = eightP.getFundamentalRANSAC(pts1, pts2)

    F_CV, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    print("Fundamental matrix from our implementation:")
    print(F)
    print("Fundamental matrix from OpenCV:")
    print(F_CV)
    print("Difference between the two matrices:")
    print(np.abs(F - F_CV))
    
    # TEST - compute the epipolar constraint error
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    error = np.abs(np.sum(pts2_hom * (F_CV @ pts1_hom.T).T, axis=1))

    # Plot epipolar lines in both images
    def plotEpipolarLines(im1, im2, pts1, pts2, F):
        # Convert images to color for visualization
        im1_color = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
        im2_color = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

        # Compute the epilines in both images
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)

        # Draw the epilines on the images
        for r, pt1, pt2 in zip(lines1, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [im1.shape[1], -(r[2] + r[0] * im1.shape[1]) / r[1]])
            im1_color = cv2.line(im1_color, (x0, y0), (x1, y1), color, 1)
            im1_color = cv2.circle(im1_color, (int(pt1[0]), int(pt1[1])), 5, color, -1)

        for r, pt1, pt2 in zip(lines2, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [im2.shape[1], -(r[2] + r[0] * im2.shape[1]) / r[1]])
            im2_color = cv2.line(im2_color, (x0, y0), (x1, y1), color, 1)
            im2_color = cv2.circle(im2_color, (int(pt2[0]), int(pt2[1])), 5, color, -1)

        # Show the images with epilines
        plt.figure(figsize=(15, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(im1_color, cv2.COLOR_BGR2RGB))
        plt.title('Epilines in Image 1'), plt.axis('off')
        plt.subplot(122), plt.imshow(cv2.cvtColor(im2_color, cv2.COLOR_BGR2RGB))
        plt.title('Epilines in Image 2'), plt.axis('off')
        plt.show()

    # Plot the epipolar lines
    # Plot the epipolar lines
    plotEpipolarLines(im3_gray, im2_gray, pts1, pts2, F)
    plotEpipolarLines(im3_gray, im2_gray, pts1, pts2, F_CV)
    
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
    R,t,_ = eightP.DEPRECATEDgetRotationTranslationFromImages(im1_gray, im2_gray, K)
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




