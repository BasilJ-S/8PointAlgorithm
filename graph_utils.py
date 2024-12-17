import cv2
import matplotlib.pyplot as plt
import numpy as np

def drawlines(img,lines,pts):
    # img - image on which we draw the epipolar lines for the points from the second image
    # lines - epipolar lines 
    r,c = img.shape
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for r,pt in zip(lines,pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
        img = cv2.circle(img,(int(pt[0]), int(pt[1])),5,color,-1)
    return img

def plot_epipolar_lines(img1, img2, pts1, pts2, F, title=''):
    # Find epipolar lines corresponding to points in second image and
    # drawing its lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2, 2,F)
    lines1 = lines1.reshape(-1,3)
    img1_epipoles= drawlines(img1, lines1, pts1)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1, 1,F)
    lines2 = lines2.reshape(-1,3)
    img2_epipoles= drawlines(img2, lines2, pts2)

    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained', sharey=True)

    ax1.imshow(img1_epipoles)
    ax1.set_title("First Image")
    ax2.imshow(img2_epipoles)
    ax2.set_title("Second Image")

    fig.suptitle('Epipolar Geometry' + title)
    fig.tight_layout(rect=[0, 0.5, 1, 0.95])
