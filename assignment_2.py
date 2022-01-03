import sys
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import reshape
import os
from operator import itemgetter
import a2_utils as util
from numpy.lib.shape_base import dsplit
np.set_printoptions(threshold=sys.maxsize)

### functions ###
def gaussdx(w, sigma):
    kernel = []
    for i in range(-w, w, 1):
        g = -1/(math.sqrt(2*math.pi)*sigma**3)*i*np.exp(-((i**2)/(2*sigma**2)))
        kernel.append(g)
    return (kernel/np.sum(np.abs(kernel)))

def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def gradient_magnitude(Ix, Iy):
    Imag = np.zeros((Ix.shape))
    Idir = np.zeros((Iy.shape))
    Imag = np.sqrt(Ix**2 + Iy**2)
    Idir = np.arctan2(Iy, Ix)
    return Imag, Idir

def ass3(I, mm):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(float)
    G = gkern(25, 0.1)
    D = -gaussdx(25, 0.1)
    Ix = cv2.filter2D(cv2.filter2D(I.copy(), -1, G.T), -1, D)
    Iy = cv2.filter2D(cv2.filter2D(I.copy(), -1, D), -1, G.T)
    Imag, Idir = gradient_magnitude(Ix, Iy)
    Idir = Idir/2
    winsize_x = math.ceil(I.shape[0]/3)
    winsize_y = math.ceil(I.shape[1]/3)
    hist_count = 0
    histograms = np.zeros((9, 8))
    for i in range(0,I.shape[0], winsize_x):
        for j in range(0, I.shape[1], winsize_y):
            window = I[i:i+winsize_x, j:j+winsize_y]
            for x in range(window.shape[0]):
                for y in range(window.shape[1]):
                    direction = Idir[i+x, j+y]
                    bin_no = int(((direction * 180 / math.pi) // 45) % 8)
                    histograms[hist_count, bin_no] += Imag[i+x, j+y]
            hist_count += 1
        
    for i in range(histograms.shape[0]):
        if(i != 0):
            histograms[0] += histograms[i]
        
    histograms[0] = histograms[0]/np.sum(histograms[0])
    return histograms[0]

def myhist3(img, n): 
    H = np.zeros((n, n, n))
    limit = math.ceil(256/n)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bin_r = int(img[i,j,0]/limit)
            bin_g = int(img[i,j,1]/limit)
            bin_b = int(img[i,j,2]/limit)
            H[bin_r, bin_g, bin_b] += + 1
    H = H/np.sum(H)
    return H

def compare_histograms(h1, h2, method):
    if(method == "L2"):
        result = np.sum(np.subtract(h1, h2) ** 2) ** 0.5
    elif(method == "chi"):
        result = 0.5 * np.sum(np.subtract(h1, h2)**2 / (h1 + h2 + np.exp(-10)))
    elif(method == "intersection"):
        result = 1 - np.sum([x for x,y in zip(h1, h2) if x < y])
    elif(method == "hellinger"):
        result = (0.5 * np.sum(np.subtract(h1**0.5, h2**0.5)**2))**0.5
    return result

def simple_convolution(I, k):
    n = math.ceil((len(k)-1)/2)
    for i in range(n, I.shape[0]-len(k), 1):
        value = I[i:i+len(k)]
        convolution.append(np.dot(value, k))

def convolution2(I, k):
    for i in range(0, I.shape[0], 1):
        if(i + len(k) > len(I)):
            break
        value = I[i:i+len(k)]
        convolution.append(np.dot(value, k))

def gaussian_kernel(sigma, x):
    kernel = []
    for i in range(-x, x, 1):
        g = 1/(math.sqrt(2*math.pi)*sigma)*np.exp(-((i**2)/(2*sigma**2)))
        kernel.append(g)
    return (kernel/np.sum(kernel))

def median(I, w):
    pos = int(w/2)
    for i in range(I.shape[0]-w):
        for j in range(I.shape[1]-w):
            kernel = I[i:i+w,j:j+w]
            value = np.median(kernel)
            I[i+pos, j+pos] = value 
    return I

def gaussfilter(I):
    kernel = gaussian_kernel(2, 15)
    I = cv2.filter2D(I, -1, kernel)
    I = I.T
    I = cv2.filter2D(I, -1, kernel)
    I = I.T
    return I

def simple_median(signal, n):
    for i in range(len(signal)-n):
        kernel = sorted(signal[i:i+n])
        value = np.median(kernel)
        signal[i + int(n/2)] = value
    return signal

def images(path, n):
    for i in os.listdir(path):
        img = cv2.imread(path+ '/' +i)
        his = ass3(img, n).reshape(-1)
        histograms.append(his)
        names.append(i)

##### TASK 1 #####
# 1a) 
img = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/images/cat1.jpg')
myhist3(img, 256)

# 1b)
img = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/images/cat1.jpg')
img2 = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/images/cat2.jpg')

n = 20
compare_histograms(myhist3(img, n), myhist3(img, n), "chi")

# 1 c)
n = 8
img1 = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/dataset/object_01_1.png')
img2 = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/dataset/object_02_1.png')
img3 = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/dataset/object_03_1.png')
h1 = myhist3(img1, n).reshape(-1)
h2 = myhist3(img2, n).reshape(-1)
h3 = myhist3(img3, n).reshape(-1)

plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.subplot(2,3,4)
plt.title(round(compare_histograms(h1, h1, "L2"), 7))
plt.plot(h1)
plt.subplot(2,3,5)
plt.title(round(compare_histograms(h1, h2, "L2"), 7))
plt.plot(h2)
plt.subplot(2,3,6)
plt.title(round(compare_histograms(h1, h3, "L2"), 7))
plt.plot(h3)
plt.show()

# 1d)
histograms = []
names = []

n = 50
path = 'C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/dataset'
distancesL2 = []
distancesCHI = []
distancesIS = []
distancesH = []
listL2 = []
listCHI = []
listIS = []
listH = []
images(path, n)
orig_image = cv2.imread(path + '/' + "object_05_1.png")
OI_his = myhist3(orig_image, n).reshape(-1)

for i in histograms:
    # distancesL2.append(compare_histograms(OI_his, i, "L2"))
    # distancesCHI.append(compare_histograms(OI_his, i, "chi"))
    # distancesIS.append(compare_histograms(OI_his, i, "intersection"))
    distancesH.append(compare_histograms(OI_his, i, "hellinger"))

for i in range(len(histograms)):
    # triple = (names[i], histograms[i], distancesL2[i])
    # listL2.append(triple)
    # triple = (names[i], histograms[i], distancesCHI[i])
    # listCHI.append(triple)
    # triple = (names[i], histograms[i], distancesIS[i])
    # listIS.append(triple)
    triple = (names[i], histograms[i], distancesH[i])
    listH.append(triple)

# sortedL2 = sorted(listL2, key=itemgetter(2))
# sortedCHI = sorted(listCHI, key=itemgetter(2))
# sortedIS = sorted(listIS, key=itemgetter(2))
sortedH = sorted(listH, key=itemgetter(2))

# sortedL2_img = [path+"/" + x[0] for x in sortedL2[:6]]
# sortedL2_hist = [x[1] for x in sortedL2[:6]]

# sortedCHI_img = [path+"/" + x[0] for x in sortedCHI[:6]]
# sortedCHI_hist = [x[1] for x in sortedCHI[:6]]

# sortedIS_img = [path+"/" + x[0] for x in sortedIS[:6]]
# sortedIS_hist = [x[1] for x in sortedIS[:6]]

sortedH_img = [path+"/" + x[0] for x in sortedH[:6]]
sortedH_hist = [x[1] for x in sortedH[:6]]

for i in range(6):
    plt.subplot(2,6,i+1)
    # img = cv2.imread(sortedL2_img[i])
    # img = cv2.imread(sortedCHI_img[i])
    # img = cv2.imread(sortedIS_img[i])
    img = cv2.imread(sortedH_img[i])
    plt.imshow(img)
    plt.subplot(2,6,i+6+1)
    # plt.plot(sortedL2_hist[i])
    # plt.plot(sortedCHI_hist[i])
    # plt.plot(sortedIS_hist[i])
    plt.plot(sortedH_hist[i])
plt.show()

# 1e)
n = 8
path = 'C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/dataset'
distances = []
orig_image = cv2.imread(path + '/' + "object_05_1.png")
OI_his = myhist3(orig_image, n).reshape(-1)
images(path, n)
for i in histograms:
    distances.append(compare_histograms(OI_his, i, "hellinger"))
plt.subplot(1,2,1)
plt.plot(distances)
indices = sorted(distances)[:6]
for i in indices:
    plt.plot(distances.index(i), i, 'ro', mfc='none')
distances.sort()
plt.subplot(1,2,2)
plt.plot(distances)
indices = distances[:6]
for i in indices:
    plt.plot(distances.index(i), i, 'ro', mfc='none')
plt.show()

# 1f)
n = 8
path = 'C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/dataset'
histograms = []
distances = []
orig_image = cv2.imread(path + '/' + "object_05_1.png")
OI_his = ass3(orig_image, n).reshape(-1)
images(path, n)
hist_sum = histograms[0]
for i in histograms[1:]:
    hist_sum += i
W = np.zeros(8)
lambda_v = 8
for i in range(len(W)):
    W[i] = np.exp(-lambda_v*hist_sum[i])
for i in histograms:
    i *= W
    i /= np.sum(i)
    distances.append(compare_histograms(OI_his, i, "hellinger"))
listHell = []
for i in range(len(histograms)):
    triple = (names[i], histograms[i], distances[i])
    listHell.append(triple)

sortedHell = sorted(listHell, key=itemgetter(2))
sortedHell_img = [path+"/" + x[0] for x in sortedHell[:6]]
sortedHell_hist = [x[1] for x in sortedHell[:6]]

for i in range(6):
    plt.subplot(2,6,i+1)
    img = cv2.imread(sortedHell_img[i])
    plt.imshow(img)
    plt.subplot(2,6,i+6+1)
    plt.plot(sortedHell_hist[i])
plt.show()

##### TASK 2 #####
# 2a)
# manually done

# 2b)
path = 'C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/'
convolution = []

signal = util.read_data(path + "signal.txt")
kernel = util.read_data(path + "kernel.txt")
simple_convolution(signal, kernel)
plt.plot(signal)
plt.plot(kernel)
plt.plot(convolution)
plt.show()

# 2c)
signal = np.array(util.read_data(path + "signal.txt"))
kernel = util.read_data(path + "kernel.txt")
signalP = np.copy(signal)
signalP = np.insert(signalP, 0, [0, 0, 0, 0, 0, 0])
signalP = np.insert(signalP, len(signalP), [0, 0, 0, 0, 0, 0])
convolution2(signalP, kernel)
plt.plot(signal)
plt.plot(kernel)
plt.plot(convolution)
plt.show()

# 2d)
plt.plot(gaussian_kernel(0.5, 15))
plt.plot(gaussian_kernel(1, 15))
plt.plot(gaussian_kernel(2, 15))
plt.plot(gaussian_kernel(3, 15))
plt.plot(gaussian_kernel(4, 15))
plt.show()

# 2e)
signal = util.read_data('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/signal.txt')
k1 = gaussian_kernel(2,15)
k2 = np.array([0.1, 0.6, 0.4])
plt.subplot(1,4,1)
plt.plot(signal)
plot1 = cv2.filter2D(cv2.filter2D(signal, -1, k1), -1, k2)
plt.subplot(1,4,2)
plt.plot(plot1)
plot2 = cv2.filter2D(cv2.filter2D(signal, -1, k2), -1, k1)
plt.subplot(1,4,3)
plt.plot(plot2)
plot3 = cv2.filter2D(signal, -1, cv2.filter2D(k1, -1, k2))
plt.subplot(1,4,4)
plt.plot(plot3)
plt.show()

##### TASK 3 #####
# 3 a)
img = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/images/lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGN = np.copy(img)
imgSP = np.copy(img)
imgGN = util.gauss_noise(imgGN, 100)
imgSP = util.sp_noise(imgSP, 0.09)
imgGN_r = gaussfilter(np.copy(imgGN))
imgSP_r = gaussfilter(np.copy(imgSP))
plt.subplot(2,2,1)
plt.imshow(imgGN, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(imgSP, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(imgGN_r, cmap='gray')
plt.subplot(2,2,4)
plt.imshow(imgSP_r, cmap='gray')
plt.show()

# 3 b)
I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/images/museum.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

kernel = np.array([[0,0,0], [0,2,0], [0,0,0]] - np.multiply(1/9, ([[1,1,1],[1,1,1],[1,1,1]])))
img = cv2.filter2D(I, -1, kernel)
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(I, cmap='gray')
plt.show()

# 3 c)
input = np.zeros(40)
input[15:25] = 1
signal = util.read_data('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/signal.txt')
signalC = np.copy(signal)
plt.subplot(1,4,1)
plt.plot(input)
plt.subplot(1,4,2)
plt.plot(signalC)
signalM = simple_median(signal, 3)
plt.subplot(1,4,3)
plt.plot(signalM)
gk = gaussian_kernel(2, 15)
signalG = cv2.filter2D(signalC, -1, gk)
plt.subplot(1,4,4)
plt.plot(signalG)
plt.show()

# 3 d)
I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_2/images/lena.png')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I_gn = util.gauss_noise(I, 100)
I_sp = util.sp_noise(I, 0.1)
plt.subplot(2,3,1)
plt.imshow(I_gn, cmap='gray')
plt.subplot(2,3,4)
plt.imshow(I_sp, cmap='gray')
img_g_gf = np.copy(I)
img_sp_gf = np.copy(I)
img_g_mf = np.copy(I)
img_sp_mf = np.copy(I)
img_g_gf = gaussfilter(I_gn)
img_sp_gf = gaussfilter(I_sp)
img_g_mf = median(I_gn, 3)
img_sp_mf = median(I_sp, 3)
plt.subplot(2,3,2)
plt.imshow(img_g_gf, cmap='gray')
plt.subplot(2,3,3)
plt.imshow(img_g_mf, cmap='gray')
plt.subplot(2,3,5)
plt.imshow(img_sp_gf, cmap='gray')
plt.subplot(2,3,6)
plt.imshow(img_sp_mf, cmap='gray')
plt.show()