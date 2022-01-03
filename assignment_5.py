import numpy as np
import cv2
import a5_utils as utils
from matplotlib import pyplot as plt
import math

#### TASK 1 ####
# 1a)

# x1 = (px/pz)*f
# x2 = -(T-px)/pz*f

# d = T*f/pz

# 1b)
# pz = 100
# f = 0.25
# T = 12
# px = 1
# distances = []
# for i in range(1,pz):
#     x1 = (px/i)*f
#     x2 = -(T-px)/i*f
#     distances.append(x1-x2)
# plt.plot(distances)
# plt.show()

# 1d)
def NCC(patch1, patch2):
    N = patch1.shape[0] * patch2.shape[1]
    std = np.std(patch1)*np.std(patch2)
    if(std != 0):
        NCC = 1/N*np.sum((patch1 - np.mean(patch1))*(patch2 - np.mean(patch2)))/std
    else:
        NCC = 0
    return NCC

def disparity(I1, I2):
    disparity_matrix = np.zeros((I1.shape[0], I1.shape[1]))
    for i in range(1, I1.shape[0] - 1):
        max_ncc = -1
        max_pixel = -1
        for j in range(1, I1.shape[1] - 1):
            patch1 = I1[i - 1: i + 1, j - 1: j + 1]
            for k in range(1, I1.shape[1]-1):
                patch2 = I2[i - 1 : i + 1, k - 1: k + 1]
                ncc = NCC(patch1, patch2)
                if(ncc > max_ncc):
                    max_ncc = ncc
                    max_pixel = k
            disparity_matrix[i, j] = max_pixel-i
    return disparity_matrix

# I1 = cv2.imread('data/disparity/slika1.png')
# I2 = cv2.imread('data/disparity/slika2.png')
# disparity_matrix1 = disparity(I1, I2)
# disparity_matrix2 = disparity(I2, I1)
# matrix = (disparity_matrix1 + disparity_matrix2)/2
# plt.imshow(matrix, cmap='gray')
# plt.show()

#### TASK 2 ####
# 2a)
def draw_oneline(F, x, I):
    x = np.append(x, [1])
    l = np.dot(F, x)
    utils.draw_epiline(l,I.shape[0],I.shape[1])

def get_epiline(F, pairs, I1, I2):
    plt.subplot(1,2,1)
    plt.imshow(I1)
    for i in range(pairs.shape[0]):
        x1 = pairs[i, 2:4]
        draw_oneline(F.T, x1, I1)
   
    plt.subplot(1,2,2)
    plt.imshow(I2)
    for i in range(pairs.shape[0]):
        x1 = pairs[i, 0:2]
        draw_oneline(F, x1, I2)
    plt.show()

def getMatrixA(points1, points2):
    A = []
    for i in range(points1.shape[0]):
        u1 = points1[i,0]
        v1 = points1[i,1]
        u2 = points2[i,0]
        v2 = points2[i,1]

        A.append([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1])
    return A

def fundamental_matrix(pairs):
    half = int(pairs.shape[1]/2)
    camera1 = pairs[:,:half]
    camera2 = pairs[:,half:pairs.shape[1]]
    points1, T1 = utils.normalize_points(camera1)
    points2, T2 = utils.normalize_points(camera2)
    A = getMatrixA(points1, points2)
    [U1, D1, V1] = np.linalg.svd(A)
    Ft = V1[-1].reshape(3,3)
    [U, D, V] = np.linalg.svd(Ft)
    D[-1] = 0
    D = np.diag(D)
    F = np.dot(np.dot(U,D), V)
    F = np.dot(np.dot(T2.T, F.T), T1)
    return F

# 2b)
# I1 = cv2.imread('data/epipolar/house1.jpg')
# I2 = cv2.imread('data/epipolar/house2.jpg')
# pairs = utils.read_data('data/epipolar/house_points.txt').reshape(10,4)
# F = fundamental_matrix(pairs)
# get_epiline(F, pairs, I1, I2)

# 2c)
def get_distance(point, params):
    distance = np.abs(params[0]*point[0] + params[1]*point[1] + params[2])/np.sqrt(params[0]**2 + params[1]**2)
    return distance

def reprojection_error(pairs, F):
    errors = []
    for i in range(pairs.shape[0]):
        point1 = pairs[i, 0:2]
        point2 = pairs[i, 2:4]
        # point1 = np.array([85, 233])
        # point2 = np.array([67, 219])
        point1 = np.append(point1, [1])
        point2 = np.append(point2, [1])
        l1 = np.dot(F.T, point2)
        l2 = np.dot(F, point1)
        distance1 = get_distance(point1, l1)
        distance2 = get_distance(point2, l2)
        avg_distance = (distance1+distance2)/2
        errors = np.append(errors, avg_distance)
    return np.sum(errors)/errors.shape[0]

# pairs = utils.read_data('data/epipolar/house_points.txt').reshape(10,4)
# F = fundamental_matrix(pairs)
# error = reprojection_error(pairs, F)
# print(error)

# 2d)
def get_inliers(F, pairs, thresh):
    inliers = []
    for i in range(pairs.shape[0]):
        x1 = np.array([pairs[i,0], pairs[i,1]])
        x2 = np.array([pairs[i,2], pairs[i,3]])
        x1 = np.append(x1, [1])
        x2 = np.append(x2, [1])
        I = np.dot(F.T, x2)
        distance = get_distance(x1, I)
        if(distance < thresh):
            inliers = np.append(inliers, [x1, x2])
    if(len(inliers) > 0):
        inliers = inliers.reshape(int(inliers.shape[0]/6), 6)
        inliers = np.delete(inliers, -1, axis=1)
        inliers = np.delete(inliers, 2, axis=1)
    return inliers

# pairs = utils.read_data('data/epipolar/house_matches.txt').reshape(168, 4)
# F = fundamental_matrix(pairs)
# inliers = get_inliers(F, pairs, 100)
# print(inliers)

# 2e)
def draw_inliers(pairs, F, I1, I2, inliers, percentage):
    points1 = np.array(pairs[0:, 0:2])
    points2 = np.array(pairs[0:, 2:4])
    green_point = np.array(pairs[1,0:2])
    green_point2 = np.array(pairs[1,2:4])
    plt.subplot(1,2,1)
    plt.imshow(I1, cmap='gray')
    for i in range(points1.shape[0]):
        point1 = np.array(points1[i])
        if(point1 in inliers[:,0:2]):
            plt.scatter(point1[0], point1[1], marker='o', color='blue')
        else:
            plt.scatter(point1[0], point1[1], marker='o', color='red')
    plt.scatter(green_point[0], green_point[1], marker='o', color='yellow')
    plt.subplot(1,2,2)
    plt.imshow(I2, cmap='gray')
    draw_oneline(F, green_point, I1)
    for i in range(points2.shape[0]):
        point2 = np.array(points2[i])
        if(point2 in inliers[:,2:4]):
            plt.scatter(point2[0], point2[1], marker='o', color='blue')
        else:
            plt.scatter(point2[0], point2[1], marker='o', color='red')
    plt.scatter(green_point2[0], green_point2[1], marker='o', color='yellow')
    error = reprojection_error(inliers, F)
    plt.title("%1.2f" %percentage + ", %1.2f" %error)
    plt.show()

def ransac_fundamental(pairs, e, k):
    F = []
    for i in range(k):
        random_ix = np.random.choice(pairs.shape[0], 8, replace=False)
        subset = pairs[random_ix, :]
        F = fundamental_matrix(subset)
        inliers = get_inliers(F, pairs, 5)
        percentage = len(inliers)/len(pairs)
        if(percentage >= e):
            F = fundamental_matrix(inliers)
            return F, inliers, percentage
    return F, inliers, percentage

# pairs = utils.read_data('data/epipolar/house_matches.txt').reshape(168, 4)
# F, inliers, percentage = ransac_fundamental(pairs, 0.70, 500)
# I1 = cv2.imread('data/epipolar/house1.jpg')
# I2 = cv2.imread('data/epipolar/house2.jpg')
# draw_inliers(pairs, F, I1, I2, inliers, percentage)

# 2f)
def simple_descriptors(I, pts, bins=150, rad=120, w=11):

	g = gauss(w, 3)
	d = gaussdx(w, 3)

	Ix = cv2.filter2D(I, cv2.CV_32F, g.T)
	Ix = cv2.filter2D(Ix, cv2.CV_32F, d)

	Iy = cv2.filter2D(I, cv2.CV_32F, g)
	Iy = cv2.filter2D(Iy, cv2.CV_32F, d.T)

	Ixx = cv2.filter2D(Ix, cv2.CV_32F, g.T)
	Ixx = cv2.filter2D(Ixx, cv2.CV_32F, d)

	Iyy = cv2.filter2D(Iy, cv2.CV_32F, g)
	Iyy = cv2.filter2D(Iyy, cv2.CV_32F, d.T)

	mag = np.sqrt(Ix**2+Iy**2)
	mag = np.floor(mag*((bins-1)/np.max(mag)))

	feat = Ixx+Iyy
	feat+=abs(np.min(feat))
	feat = np.floor(feat*((bins-1)/np.max(feat)))

	desc = []

	for y,x in pts:
		minx = max(x-rad, 0)
		maxx = min(x+rad, I.shape[0])
		miny = max(y-rad, 0)
		maxy = min(y+rad, I.shape[1])
		r1 = mag[minx:maxx, miny:maxy].reshape(-1)
		r2 = feat[minx:maxx, miny:maxy].reshape(-1)
	
		a = np.zeros((bins,bins))
		for m, l in zip(r1,r2):
			a[int(m),int(l)]+=1

		a=a.reshape(-1)
		a/=np.sum(a)

		desc.append(a)

	return np.array(desc)

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

def display_matches(im1, im2, pts1, pts2, matches):

	# NOTE: this will only work correctly for images with the same height
	# NOTE: matches should contain index pairs (i.e. first element is the index to pts1 and second for pts2)

	I = np.hstack((im1,im2))
	w = im1.shape[1]
	plt.clf()
	plt.imshow(I, cmap='gray')

	for i, j in matches:
		p1 = pts1[int(i)]
		p2 = pts2[int(j)]
		plt.plot(p1[0], p1[1], 'bo')
		plt.plot(p2[0]+w, p2[1], 'bo')
		plt.plot([p1[0], p2[0]+w], [p1[1], p2[1]], 'r')

	plt.draw()
	#plt.waitforbuttonpress()
	plt.show()

def find_matches(I, I2):
    detector = cv2.ORB_create()
    points = detector.detect(I, None)
    points2 = detector.detect(I2, None)
    points, des1 = detector.compute(I, points)
    points2, des2 = detector.compute(I2, points2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:50]
    pts1 = []
    pts2 = []
    for match in matches:
        pts1 = np.append(pts1, np.array(points[match.queryIdx].pt))
        pts2 = np.append(pts2, np.array(points2[match.trainIdx].pt))

    pts1 = pts1.reshape(int(pts1.shape[0]/2), 2)
    pts2 = pts2.reshape(int(pts2.shape[0]/2), 2)
    matches = np.concatenate((pts1, pts2), axis=1)
    return matches, pts1, pts2

# I1 = cv2.cvtColor(cv2.imread('data/epipolar/house1.jpg'), cv2.COLOR_BGR2GRAY)
# I2 = cv2.cvtColor(cv2.imread('data/epipolar/house2.jpg'), cv2.COLOR_BGR2GRAY)
# matches, points, points2 = find_matches(I1, I2)
# pairs = np.concatenate((points, points2), axis=1)
# F, inliers, percentage = ransac_fundamental(pairs, 0.75, 100)
# draw_inliers(pairs, F, I1, I2, inliers, percentage)

#### TASK 3 ####
# 3a)
def create_matrix(c, cm):
    x = np.zeros((3,3))
    x[0,1] = -1
    x[0,2] = c[1]
    x[1,0] = 1
    x[1,2] = -c[0]
    x[2,0] = -c[1]
    x[2,1] = c[0]

    return np.dot(x, cm)[:2]

def triangulate(pairs, cal_mat1, cal_mat2):
    res = []
    for i in range(pairs.shape[0]):
        x1 = pairs[i, 0:2]
        A1 = create_matrix(x1, cal_mat1)
        x2 = pairs[i, 2:4]
        A2 = create_matrix(x2, cal_mat2)
        A = np.bmat([[A1], [A2]])
        [U, D, V] = np.linalg.svd(A)
        lowestEigenVector = V[-1]
        lowestEigenVector = lowestEigenVector/lowestEigenVector[-1, -1]
        res = np.append(res, lowestEigenVector)
    return res

# pairs = utils.read_data('data/epipolar/house_points.txt').reshape(10,4)
# calibrate_matrix1 = utils.read_data('data/epipolar/house1_camera.txt').reshape(3,4)
# calibrate_matrix2 = utils.read_data('data/epipolar/house2_camera.txt').reshape(3,4)
# res = triangulate(pairs, calibrate_matrix1, calibrate_matrix2).reshape(10,4)
# res = np.delete(res, -1, axis=1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d') # define 3D subplot
# T = np.array([[-1,0,0],[0,0,1],[0,-1,0]]) # transformation matrix
# res = np.dot(res,T)
# for i, pt in enumerate(res):
#     plt.plot([pt[0]],[pt[1]],[pt[2]],'r.') # plot points
#     ax.text(pt[0],pt[1],pt[2], str(i)) # plot indices
# plt.show()