from typing import Counter
from PIL.Image import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from a6_utils import drawEllipse
import os

def read_data(filename):
    # reads a numpy array from a text file
    with open(filename) as f:
        s = f.read()

    return np.fromstring(s, sep=' ')

#### TASK 1 ####
# 1a)
def PCA(data, showEllipse):
    mean = np.mean(data, axis = 0)
    data = data - mean
    cov = np.cov(data, rowvar = False)
    if(showEllipse):
        drawEllipse(mean, cov)
    [U, S, V] = np.linalg.svd(cov)
    return U, S, mean

data = read_data('data/points.txt').reshape(5, 2)
for i in range(data.shape[0]):
   plt.scatter(data[i,0], data[i,1], marker = "o", color="blue") 
PCA(data, 1)
plt.show()

# 1b)
data = read_data('data/points.txt').reshape(5,2)
vectors, values, mean = PCA(data.copy(), 1)
vector1 = vectors[:,0] * values[0]
vector2 = vectors[:,1] * values[1]
for i in range(data.shape[0]):
    plt.scatter(data[i,0], data[i,1], marker = "o", color="blue")
plt.plot([mean[0],mean[0]+vector1[0]],[mean[1],mean[1]+vector1[1]])
plt.plot([mean[0],mean[0]+vector2[0]],[mean[1],mean[1]+vector2[1]])
plt.show()

# 1c)
def reconstruction_error(values):
    c_value = []
    value = 0
    for i in range(len(values)):
        value += values[i]
        c_value = np.append(c_value, value)
    c_value = c_value/c_value[-1]
    plt.bar(np.arange(len(values)), c_value)
    plt.show()

data = read_data('data/points.txt').reshape(5,2)
vectors, values, mean = PCA(data.copy(), 0)
reconstruction_error(values)

# 1d)
data = read_data('data/points.txt').reshape(5,2)
vectors, values, mean = PCA(data.copy(), 1)
vectors[:,1] = 0
vector1 = vectors[:,0] * values[0]
point = data.copy()
for i in range(data.shape[0]):
    point[i] = vectors.T @ (point[i] - mean)
    point[i] = vectors @ point[i] + mean
    print(point[i])
    plt.scatter(point[i,0], point[i,1], marker = "o", color="blue")
plt.plot([mean[0],mean[0]+vector1[0]],[mean[1],mean[1]+vector1[1]])
plt.show()

# 1e)
def get_distance(data, point):
    closest_point = np.zeros((1,1))
    min_distance = 1000
    for i in range(data.shape[0]):
        distance = np.linalg.norm(data[i,:]-point)
        if(distance < min_distance):
            min_distance = distance
            closest_point = data[i,:]
    return closest_point

point = np.array([3, 6])
data = read_data('data/points.txt').reshape(5,2)
first_point = get_distance(data, point)
data = np.append(data, [point.T]).reshape(6,2)
vectors, values, mean = PCA(data.copy(), 0)
vectors[0,:] = 0
points = data.copy()
for i in range(data.shape[0]):
    points[i] = vectors.T  @ (points[i]-mean)
    points[i] = vectors @ points[i] + mean
points = points[:5,:]
index = get_distance(points, point)
indexOG = np.where((points == (index)).all(axis=1))
print(data[indexOG])
print(first_point)

#### TASK 2 ####

# 2a)
def dualPCA(data):
    mean = np.mean(data, axis = 0)
    data = data - mean
    C = (1/(data.shape[1]-1))*np.dot(data, data.T)
    [U, S, V] = np.linalg.svd(C)
    U = np.dot(data.T, U) * np.sqrt(1 / (S*(data.shape[1]-1)))
    return U, S, mean

data = read_data('data/points.txt').reshape(5,2)
vectors, values, mean = dualPCA(data)
vector1 = vectors[:,0] * values[0]
vector2 = vectors[:,1] * values[1]
for i in range(data.shape[0]):
    plt.scatter(data[i,0], data[i,1], marker = "o", color="blue")
plt.plot([mean[0],mean[0]+vector1[0]],[mean[1],mean[1]+vector1[1]])
plt.plot([mean[0],mean[0]+vector2[0]],[mean[1],mean[1]+vector2[1]])
plt.show()

# # 2b)
data = read_data('data/points.txt').reshape(5, 2)
U, S, mean = dualPCA(data.copy())
points = data.copy()
U = U[:,:2]
for i in range(data.shape[0]):
    points[i] = np.dot(U.T, (points[i]-mean))
for i in range(data.shape[0]):
    points[i] = (np.dot(U, points[i]) + mean)
print(points)

#### TASK 3 ####
# 3a)
def dualPCA_img(data):
    mean = np.average(data, axis = 1)
    for i in range(data.shape[1]):
        data[:, i] = data[:, i] - mean
    C = (1/(data.shape[1]-1))*np.dot(data.T, data)
    [U, S, V] = np.linalg.svd(C)
    U = np.dot(data, U) * np.sqrt(1 / (S*(data.shape[1]-1)))
    return U, S, mean

def preprocess(folder):
    data = np.zeros((8064, 64))
    counter = 0
    for filename in os.listdir(folder):
        img = cv2.cvtColor(cv2.imread(os.path.join(folder,filename)), cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1)
        data[:, counter] = img
        counter += 1
    return data

def firstFive(U):
    firstFive = U[:,0:5]
    for i in range(firstFive.shape[1]):
        image = firstFive[:,i].reshape((96, 84))
        plt.subplot(1,5,i+1)
        plt.imshow(image, cmap='gray')
    plt.show()

def transformations(image, mean, U):
    imageA = image.copy()
    imageB = image.copy()
    imageC = image.copy()

    imageA = U.T @ (imageA - mean)
    imageA = U @ imageA + mean
    imageA_diff = image - imageA
    plt.subplot(1,3,1)
    plt.imshow(imageA.reshape((96, 84)), cmap='gray', vmax=255, vmin=0)

    imageB[4074] = 0
    imageB = U.T @ (imageB - mean)
    imageB = U @ imageB + mean
    imageB_diff = image - imageB
    plt.subplot(1,3,2)
    plt.imshow(imageB.reshape((96, 84)), cmap='gray', vmax=255, vmin=0)

    imageC = U.T @ (imageC - mean)
    imageC[1] = 0
    imageC = U @ imageC + mean
    imageC_diff = image - imageC
    plt.subplot(1,3,3)
    plt.imshow(imageC.reshape((96, 84)), cmap='gray', vmax=255, vmin=0)
    plt.show()

# 3b)
folder = 'data/faces/1'
data = preprocess(folder)
U, S, mean = dualPCA_img(data.copy())
firstFive(U)
transformations(data[:,0], mean, U)

# 3c) 
def reconstruction(n, image, U, mean):
    image = U.T @ (image-mean)
    image[n:] = 0
    image = U @ image+mean
    image = image.reshape((96, 84))
    return image

def num_of_comp(image, U, mean):
    for i in range(6):
        img = reconstruction(int(32/2**i), image, U.copy(), mean.copy())
        plt.subplot(1,6,i+1)
        plt.imshow(img, cmap='gray', vmax=255, vmin=0)
    plt.show()

folder = 'data/faces/2'
data = preprocess(folder)
U, S, mean = dualPCA_img(data.copy())
num_of_comp(data[:,0], U, mean)

# 3 d)
def informativeness(avg_img, U, mean):
    change1 = np.sin(np.linspace(-10, 10))*3000
    change2 = np.cos(np.linspace(-10, 10))*3000
    for i in range(10):
        avg_img[0] = change1[i]
        avg_img[4] = change2[i]
        show_img = U.T @ (avg_img - mean)
        show_img = U @ avg_img + mean
        plt.subplot(2,5,i+1)
        plt.imshow(show_img.reshape((96,84)), cmap='gray')
    
    plt.show()

folder = 'data/faces/2'
data = preprocess(folder)
U, S, mean = dualPCA_img(data.copy())
avg_img = U.T @ (mean-mean)
informativeness(avg_img.copy(), U, mean)

# 3 e)
def transform_foreign(image, U, mean):
    image = U.T @ (image - mean)
    image = U @ image + mean
    image = image.reshape((96, 84))
    return image

folder = 'data/faces/1'
data = preprocess(folder)
U, S, mean = dualPCA_img(data.copy())
foreign_img = cv2.imread('data/elephant.jpg')
foreign_img = foreign_img[:, :, 0]
img = transform_foreign(foreign_img.copy().reshape(-1), U, mean)
plt.subplot(1,2,1)
plt.imshow(foreign_img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

# 3g)
def lda(data):
    sb = 0
    sw = 0
    mm = np.mean(data, axis=1, keepdims=True)
    ms = np.zeros((10,3))
    c = 3
    n = 64
    for i in range(c):
        ms[:,i] = np.mean(data[:,i*n:(i+1)*n], axis=1)
        sb = sb + n * (ms[:,[i]]-mm) @ (ms[:,[i]]-mm).T
        for j in range(n):
            sw = sw + (data[:,[i*n+j]] - ms[:,[i]]) @ (data[:,[i*n+j]]-ms[:,[i]]).T
    [U, S, V] = np.linalg.svd(np.linalg.inv(sw) @ sb)
    return U

data = np.zeros((8064, 64*3))

for i in range(1,4,1):
    folder = 'data/faces/'+ str(i)
    imgs = preprocess(folder)
    data[:, (64*(i-1)):(64*i)] = imgs

data = data/np.max(data)
U, S, mean = dualPCA_img(data.copy())
mean = mean.reshape((-1,1))
dataPCA = U.T @ (data - mean)
plt.subplot(1,2,1)
for i in range(data.shape[1]):
    if(i < 64):
        plt.scatter(dataPCA[0,i], dataPCA[1,i], marker = "o", color="blue")
    elif(i >= 64 and i < 128 ):
        plt.scatter(dataPCA[0,i], dataPCA[1,i], marker = "o", color="red")
    else:
        plt.scatter(dataPCA[0,i], dataPCA[1,i], marker = "o", color="green")

mean2 = np.mean(dataPCA, axis=1, keepdims=True)
U_lda = lda(dataPCA[:10,:].copy())
dataLDA = U_lda.T @ (dataPCA - mean2)[:10]
plt.subplot(1,2,2)
for i in range(data.shape[1]):
    if(i < 64):
        plt.scatter(dataLDA[0,i], dataLDA[1,i], marker = "o", color="blue")
    elif(i >= 64 and i < 128 ):
        plt.scatter(dataLDA[0,i], dataLDA[1,i], marker = "o", color="red")
    else:
        plt.scatter(dataLDA[0,i], dataLDA[1,i], marker = "o", color="green")
plt.show()