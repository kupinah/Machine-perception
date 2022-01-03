import sys
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import a4_utils as utils
#np.set_printoptions(threshold=sys.maxsize)

### TASK 1 ###
# 1 a)
def gaussfilter(I):
    kernel = gauss(10, 3)
    I = cv2.filter2D(I, -1, kernel)
    I = I.T
    I = cv2.filter2D(I, -1, kernel)
    I = I.T
    return I

def gaussdx(w, sigma):
    kernel = []
    for i in range(-w, w, 1):
        g = -1/(math.sqrt(2*math.pi)*sigma**3)*i*np.exp(-((i**2)/(2*sigma**2)))
        kernel.append(g)
    return kernel/np.sum(np.abs(kernel))

def gauss(x, sigma):
    kernel = []
    for i in range(-x, x, 1):
        g = 1/(math.sqrt(2*math.pi)*sigma)*np.exp(-((i**2)/(2*sigma**2)))
        kernel.append(g)
    return kernel/np.sum(kernel)

def nms(I, thresh):
    x_cord = []
    y_cord = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if(I[i, j] > thresh):
                left = max(0,i-1)
                right = max(0,i+1+1)
                bottom = max(0,j-1)
                top = max(0,j+1+1)
                neigh = I[left:right,bottom:top]
                add_flag = 1
                for m in range(neigh.shape[0]):
                    for n in range(neigh.shape[1]):
                        if(m == 1 and n == 1):
                            continue
                        if(I[i, j] <= neigh[m, n]):
                            add_flag = 0
                if(add_flag == 1):
                    x_cord.append(i)
                    y_cord.append(j)
    return x_cord, y_cord

def hessian_point(I, sigma):
    G = gauss(10, sigma)
    D = -gaussdx(10, sigma)

    Ix = cv2.filter2D(cv2.filter2D(I.copy(), -1, G).T, -1, D).T
    Iy = cv2.filter2D(cv2.filter2D(I.copy(), -1, D).T, -1, G).T
   
    Ixx = cv2.filter2D(cv2.filter2D(Ix.copy(), -1, G).T, -1, D).T
    Ixy = cv2.filter2D(cv2.filter2D(Ix.copy(), -1, D).T, -1, G).T
    Iyy = cv2.filter2D(cv2.filter2D(Iy.copy(), -1, D).T, -1, G).T
    
    det = sigma**4*(Ixx*Iyy-Ixy**2)
    return det

# I = cv2.cvtColor(cv2.imread('data/graf/graf1.jpg'), cv2.COLOR_BGR2GRAY).astype(float)
# Is = gaussfilter(I.copy())
# hessian = hessian_point(Is.copy(), 3)
# x_cord, y_cord = nms(hessian.copy(), 15000)
# plt.subplot(1,2,1)
# plt.imshow(hessian, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(I, cmap='gray')
# plt.scatter(y_cord, x_cord, marker='x', color='red')
# plt.show()

# 1b)
def nms_harris(I, thresh, det, trace):
    x_cord = []
    y_cord = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            value = det[i,j] - 0.06*trace[i,j]
            if(value > thresh):
                left = max(0,i-1)
                right = max(0,i+1+1)
                bottom = max(0,j-1)
                top = max(0,j+1+1)
                neighDet = det[left:right,bottom:top]
                neighTrace = trace[left:right,bottom:top]
                neigh = neighDet - 0.06*neighTrace
                add_flag = 1
                for m in range(neigh.shape[0]):
                    for n in range(neigh.shape[1]):
                        if(m == 1 and n == 1):
                            continue
                        n_value = neigh[m, n]
                        if(value <= n_value):
                            add_flag = 0
                if(add_flag == 1):
                    x_cord.append(i)
                    y_cord.append(j)
    return x_cord, y_cord

def harris_point(I, sigma):
    sigma2 = 1.6*sigma
    g = gauss(10, sigma)
    d = -gaussdx(10, sigma)
    Gk = gauss(10, sigma2)
    G = cv2.filter2D(I.copy(), -1, Gk) 

    Ix = cv2.filter2D(cv2.filter2D(I.copy(), -1, g).T, -1, d).T
    Iy = cv2.filter2D(cv2.filter2D(I.copy(), -1, d).T, -1, g).T
    ele11 = G*Ix**2
    ele12 = G*Ix*Iy
    ele22 = G*Iy**2

    C = sigma**2*(np.bmat([[ele11, ele12], [ele12, ele22]]))
    det = ele11*ele12 - ele12*ele22
    trace = ele11+ele22
    return det, trace

# I = cv2.cvtColor(cv2.imread('data/graf/graf1.jpg'), cv2.COLOR_BGR2GRAY).astype(float)
# Is = gaussfilter(I.copy())
# det, trace = harris_point(Is.copy(), 3)
# x_cord, y_cord = nms_harris(Is.copy(), 100000000, det, trace**2)
# plt.imshow(I, cmap='gray')
# plt.scatter(y_cord, x_cord, marker='x', color='blue')
# plt.show()

#### TASK 2 ####
# 2a

def hellingers_distance(x, y):
    result = (0.5 * np.sum(np.subtract(x**0.5, y**0.5)**2))**0.5
    return result

def find_correspondences(d1, d2):
    most_similar = []
    for i in range(len(d1)):
        min_v = 100000
        min_j = -1
        for j in range(len(d2)):
            distance = hellingers_distance(d1[i], d2[j])
            if(distance < min_v):
                min_v = distance
                min_j = j
        most_similar.append((i, min_j))
    return most_similar

# 2b
def nms2(I, thresh):
    coordinates = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if(I[i, j] > thresh):
                left = max(0,i-1)
                right = max(0,i+1+1)
                bottom = max(0,j-1)
                top = max(0,j+1+1)
                neigh = I[left:right,bottom:top]
                add_flag = 1
                for m in range(neigh.shape[0]):
                    for n in range(neigh.shape[1]):
                        if(m == 1 and n == 1):
                            continue
                        if(I[i, j] <= neigh[m, n]):
                            add_flag = 0
                if(add_flag == 1):
                    coordinates.append((j, i))
    return coordinates

# I = cv2.cvtColor(cv2.imread('data/graf/graf1.jpg'), cv2.COLOR_BGR2GRAY)
# I2 = cv2.cvtColor(cv2.imread('data/graf/graf2.jpg'), cv2.COLOR_BGR2GRAY)
# points = nms2(hessian_point(I, 3), 254)
# points2 = nms2(hessian_point(I2, 3), 254)
# descriptors = utils.simple_descriptors(I, points)
# descriptors2 = utils.simple_descriptors(I2, points2)
# matches = find_correspondences(descriptors, descriptors2)
# utils.display_matches(I, I2, points, points2, matches)

# 2c)
def find_matches(I, I2):
    points = nms2(hessian_point(I, 3), 254)
    points2 = nms2(hessian_point(I2, 3), 254)

    descriptors = utils.simple_descriptors(I, points)
    descriptors2 = utils.simple_descriptors(I2, points2)

    matches1 = find_correspondences(descriptors, descriptors2)
    matches2 = find_correspondences(descriptors2, descriptors)
    matches2 = [(t[1], t[0]) for t in matches2]

    matches = set(matches1).intersection(matches2)
    return matches, points, points2

# I = cv2.cvtColor(cv2.imread('data/graf/graf1.jpg'), cv2.COLOR_BGR2GRAY)
# I2 = cv2.cvtColor(cv2.imread('data/graf/graf2.jpg'), cv2.COLOR_BGR2GRAY)
# matches, points, points2 = find_matches(I, I2)
# utils.display_matches(I, I2, points, points2, matches)

# 2d)
### DOLGO RAČUNA, SHRANJENA SLIKA -> PRILAGODI PARAMETRE (3, 220) ter SPREMENI KLIC find_corr2 v find_matches!!! ###
def find_correspondences2(d1, d2):
    distances = []
    most_similar = []
    for i in range(len(d1)):
        for j in range(len(d2)):
            distance = hellingers_distance(d1[i], d2[j])
            dist_arr = (i, j, distance)
            distances.append(dist_arr)
        distances = sorted(distances, key=lambda x: x[2])
        distances = distances[0:2]
        j1, j2 = ([x[1] for x in distances])
        first, second = ([x[2] for x in distances])
        if(first/second >= 0.8):
            most_similar.append((i, j1))
    return most_similar

# I = cv2.cvtColor(cv2.imread('data/graf/graf1.jpg'), cv2.COLOR_BGR2GRAY)
# I2 = cv2.cvtColor(cv2.imread('data/graf/graf2.jpg'), cv2.COLOR_BGR2GRAY)
# matches, points, points2 = find_matches(I, I2)
# utils.display_matches(I, I2, points, points2, matches)

# 2f)
# video = cv2.VideoCapture('data/video.mp4')
# while(video.isOpened()):
#     ret, frame = video.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if ret == True:
#         detector = cv2.FastFeatureDetector_create(20)
#         keypoints = detector.detect(frame, None)
#         img = cv2.drawKeypoints(frame, keypoints, None)
#         cv2.imshow('FAST', img)
#         cv2.waitKey()
#     else:
#         break

##### TASK 3 #####
# 3a)
def estimate_homography(pairs):
    half = int(len(pairs)/2)
    pts1 = pairs[:,:half]
    pts2 = pairs[:,half:len(pairs)+1]
    #matches = [(0,0),(1,1),(2,2),(3,3)]
    #utils.display_matches(I, I2, pts1, pts2, matches)
    A = []
    for i in range(len(pts1)):
        A.append([pts1[i,0], pts1[i,1], 1, 0, 0, 0, -pts1[i,0]*pts2[i,0], -pts2[i,0]*pts1[i,1], -pts2[i,0]])
        A.append([0, 0, 0, pts1[i,0], pts1[i,1], 1, -pts1[i,0]*pts2[i,1], -pts1[i,1]*pts2[i,1], -pts2[i,1]])
    [U, S, V] = np.linalg.svd(A)
    V = V.T
    h = V[:,-1]
    H = (h/h[-1]).reshape(3,3)
    return H

# 3b)
# I = cv2.cvtColor(cv2.imread('data/newyork/newyork1.jpg'), cv2.COLOR_BGR2GRAY)
# I2 = cv2.cvtColor(cv2.imread('data/newyork/newyork2.jpg'), cv2.COLOR_BGR2GRAY)
# pairs = utils.read_data('data/newyork/newyork.txt')
# pairs = pairs.reshape(-1, 4)
# H = estimate_homography(pairs)
# H = cv2.warpPerspective(src=I2,M=H,dsize=(250,250))
# plt.imshow(H, cmap='gray')
# plt.show()

# 3c)
def find_correspondences3(d1, d2):
    distances = []
    most_similar = []
    for i in range(len(d1)):
        for j in range(len(d2)):
            distance = hellingers_distance(d1[i], d2[j])
            dist_arr = (i, j, distance)
            distances.append(dist_arr)
        distances = sorted(distances, key=lambda x: x[2])
        distances = distances[0:2]
        j1, j2 = ([x[1] for x in distances])
        first, second = ([x[2] for x in distances])
        if(first/second >= 0.8):
            most_similar.append((i, j1, distances[0]))
    
    return most_similar

def find_matches2(I, I2):
    points = nms2(hessian_point(I, 3), 249)
    points2 = nms2(hessian_point(I2, 3), 249)

    descriptors = utils.simple_descriptors(I, points)
    descriptors2 = utils.simple_descriptors(I2, points2)

    matches1 = find_correspondences3(descriptors, descriptors2)
    matches2 = find_correspondences3(descriptors2, descriptors)
    matches2 = [(t[1], t[0], t[2]) for t in matches2]

    matches1 = sorted(matches1, key=lambda x: x[2])
    matches2 = sorted(matches2, key=lambda x: x[2])

    matches1 = [(x,y) for x,y,z in matches1]
    matches2 = [(x,y) for x,y,z in matches2]
    matches = set(matches1).intersection(matches2)

    return matches, points, points2

def estimate_homography2(pairs, pts1, pts2):
    A = []
    for i in range(len(pts1)):
        A.append([pts1[i,0], pts1[i,1], 1, 0, 0, 0, -pts1[i,0]*pts2[i,0], -pts2[i,0]*pts1[i,1], -pts2[i,0]])
        A.append([0, 0, 0, pts1[i,0], pts1[i,1], 1, -pts1[i,0]*pts2[i,1], -pts1[i,1]*pts2[i,1], -pts2[i,1]])
    [U, S, V] = np.linalg.svd(A)
    V = V.T
    h = V[:,-1]
    H = (h/h[-1]).reshape(3,3)
    return H

# I = cv2.cvtColor(cv2.imread('data/graf/graf1.jpg'), cv2.COLOR_BGR2GRAY)
# I2 = cv2.cvtColor(cv2.imread('data/graf/graf2.jpg'), cv2.COLOR_BGR2GRAY)

# matches, pts, pts2 = find_matches2(I, I2)
# pts1_ix = [x[0] for x in matches]
# pts2_ix = [x[1] for x in matches]
# pts = np.array([pts[x] for x in pts1_ix])
# pts2 = np.array([pts2[x] for x in pts2_ix])
# H = estimate_homography2(matches, pts, pts2)
# H = cv2.warpPerspective(src=I2,M=H,dsize=(800,640))
# plt.imshow(H, cmap='gray')
# plt.show()

# 3e)
def postprocess(I):
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if(I[i, j] == 0):
                left = max(0,i-1)
                right = max(0,i+1+1)
                bottom = max(0,j-1)
                top = max(0,j+1+1)
                neigh = I[left:right,bottom:top]
                avg_neigh = np.average(neigh)
                I[i, j] = avg_neigh
    return I

def warpPerspective(I, H):
    I2 = np.zeros((I.shape[0], I.shape[1]))
    for i in range(I.shape[1]):
        for j in range(I.shape[0]):
            orig_indices = [i,j,1]
            xt = np.dot(H, orig_indices)
            xt = np.where(xt < 0, 0, xt)
            if(xt[0] > I2.shape[0] or xt[1] > I2.shape[1]):
                continue
            I2[math.floor(xt[1]), math.floor(xt[0])] = I[j, i]
    return I2

I = cv2.cvtColor(cv2.imread('data/newyork/newyork1.jpg'), cv2.COLOR_BGR2GRAY)
pairs = utils.read_data('data/newyork/newyork.txt')
pairs = pairs.reshape(-1, 4)
H = estimate_homography(pairs)
I2 = warpPerspective(I, H)
I2 = postprocess(I2)
plt.imshow(I2, cmap='gray')
plt.show()