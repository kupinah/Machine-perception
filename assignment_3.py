import sys
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import a3_utils as util
np.set_printoptions(threshold=sys.maxsize)
### TASK 1 ###
# 1a)
# done manually

# 1b)
def gaussdx(w, sigma):
    kernel = []
    for i in range(-w, w, 1):
        g = -1/(math.sqrt(2*math.pi)*sigma**3)*i*np.exp(-((i**2)/(2*sigma**2)))
        kernel.append(g)
    return (kernel/np.sum(np.abs(kernel)))

# plt.plot(gaussdx(15, 0.5))
# plt.plot(gaussdx(15, 1))
# plt.plot(gaussdx(15, 2))
# plt.plot(gaussdx(15, 3))
# plt.plot(gaussdx(15, 4))
# plt.show()

# 1c)
def gaussian_kernel(sigma, x):
    kernel = []
    for i in range(-x, x, 1):
        g = 1/(math.sqrt(2*math.pi)*sigma)*np.exp(-((i**2)/(2*sigma**2)))
        kernel.append(g)
    return (kernel/np.sum(kernel))

# impulse = np.zeros((25,25))
# impulse[12, 12] = 255
# G = gaussian_kernel(1, 20)
# D = -gaussdx(10, 1)
# plt.subplot(2,3,1)
# plt.title("Impulse")
# plt.imshow(impulse, cmap='gray')
# plt.subplot(2,3,2)
# plt.title("G, Dt")
# GDT = cv2.filter2D(cv2.filter2D(impulse.copy(), -1, G).T, -1, D)
# plt.imshow(GDT, cmap='gray')
# plt.subplot(2,3,3)
# plt.title("D, Gt")
# DGT = cv2.filter2D(cv2.filter2D(impulse.copy(), -1, D).T, -1, G)
# plt.imshow(DGT, cmap='gray')
# plt.subplot(2,3,4)
# plt.title("G, Gt")
# GGT = cv2.filter2D(cv2.filter2D(impulse.copy(), -1, G).T, -1, G).T
# plt.imshow(GGT, cmap='gray')
# plt.subplot(2,3,5)
# plt.title("Gt, D")
# GTD = cv2.filter2D(cv2.filter2D(impulse.copy(), -1, G).T, -1, D).T
# plt.imshow(GTD, cmap='gray')
# plt.subplot(2,3,6)
# plt.title("Dt, G")
# DTG = cv2.filter2D(cv2.filter2D(impulse.copy(), -1, D).T, -1, G).T
# plt.imshow(DTG, cmap='gray')
# plt.show()

# 1d)
def gradient_magnitude(Ix, Iy):
    Imag = np.zeros((Ix.shape))
    Idir = np.zeros((Iy.shape))
    for i in range(Ix.shape[0]):
        for j in range(Ix.shape[1]):
            Imag[i,j] = np.sqrt(Ix[i,j]**2 + Iy[i,j]**2)
            Idir[i,j] = np.arctan2(Iy[i,j], Ix[i,j])
    return Imag, Idir

# img = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_3/images/museum.jpg') 
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.astype(float)
G = gaussian_kernel(1, 10)
D = -gaussdx(10, 1)
Ix = cv2.filter2D(cv2.filter2D(img.copy(), -1, G).T, -1, D).T
Iy = cv2.filter2D(cv2.filter2D(img.copy(), -1, D).T, -1, G).T
Ixx = cv2.filter2D(cv2.filter2D(Ix.copy(), -1, G).T, -1, D).T
Ixy = cv2.filter2D(cv2.filter2D(Ix.copy(), -1, D).T, -1, G).T
Iyy = cv2.filter2D(cv2.filter2D(Iy.copy(), -1, D).T, -1, G).T
# Imag, Idir = gradient_magnitude(Ix, Iy)
# plt.subplot(2,4,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(2,4,2)
# plt.imshow(Ix, cmap='gray')
# plt.subplot(2,4,3)
# plt.imshow(Iy, cmap='gray')
# plt.subplot(2,4,5)
# plt.imshow(Ixx, cmap='gray')
# plt.subplot(2,4,6)
# plt.imshow(Ixy, cmap='gray')
# plt.subplot(2,4,7)
# plt.imshow(Iyy, cmap='gray')
# plt.subplot(2,4,4)
# plt.imshow(Imag, cmap='gray')
# plt.subplot(2,4,8)
# plt.imshow(Idir, cmap='gray')
# plt.show()

# 1 e)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(float)
# G = gkern(25, 0.1)
# D = -gaussdx(25, 0.1)
# Ix = cv2.filter2D(cv2.filter2D(I.copy(), -1, G.T), -1, D)
# Iy = cv2.filter2D(cv2.filter2D(I.copy(), -1, D), -1, G.T)
# Imag, Idir = gradient_magnitude(Ix, Iy)
# Idir = Idir/2
# winsize_x = math.ceil(I.shape[0]/3)
# winsize_y = math.ceil(I.shape[1]/3)
# hist_count = 0
# histograms = np.zeros((9, 8))
# for i in range(0,I.shape[0], winsize_x):
#     for j in range(0, I.shape[1], winsize_y):
#         window = I[i:i+winsize_x, j:j+winsize_y]
#         for x in range(window.shape[0]):
#             for y in range(window.shape[1]):
#                 direction = Idir[i+x, j+y]
#                 bin_no = int(((direction * 180 / math.pi) // 45) % 8)
#                 histograms[hist_count, bin_no] += Imag[i+x, j+y]
#         hist_count += 1
    
# for i in range(histograms.shape[0]):
#     if(i != 0):
#         histograms[0] += histograms[i]
    
# histograms[0] = histograms[0]/np.sum(histograms[0])
# return histograms[0]         

### TASK 2 ###
# 2a)
def findeges(img, sigma, theta):
    G = gaussian_kernel(sigma, 25)
    D = -gaussdx(25, sigma)
    Ix = cv2.filter2D(cv2.filter2D(img.copy(), -1, G).T, -1, D).T
    Iy = cv2.filter2D(cv2.filter2D(img.copy(), -1, D).T, -1, G).T
    Imag, Idir = gradient_magnitude(Ix, Iy)
    img = np.where(Imag >= theta, 1, 0)
    return img, Imag, Idir

# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_3/images/museum.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# I = I.astype(float)
# plt.imshow(findeges(I, 0.1, 30)[0], cmap='gray')
# plt.show()

# 2b)
def nms(Iorig, Imag, Idir):
    for i in range(1, Imag.shape[0]-1):
        for j in range(1,Imag.shape[1]-1):
            direction = Idir[i, j]
            if ((math.pi/5 <= direction <= 2/5*math.pi) or (-4/5*math.pi <= direction <= -3/5*math.pi)):
                if(Imag[i, j] <= Imag[i+1, j-1] and Imag[i, j] <= Imag[i-1, j+1]):
                    Iorig[i, j] = 0
            elif((2/5*math.pi <= direction <= 3/5*math.pi) or (-3/5*math.pi <= direction <= -2/5*math.pi)):
                if(Imag[i, j] <= Imag[i+1][j] and Imag[i, j] <= Imag[i-1, j]):
                    Iorig[i, j] = 0
            elif ((3/5*math.pi <= direction <= 4/5*math.pi) or (-2/5*math.pi <= direction <= -1/5*math.pi)):
                if(Imag[i, j] <= Imag[i+1, j+1] and Imag[i, j] <= Imag[i-1, j-1]):
                    Iorig[i, j] = 0
            else:
                if(Imag[i, j] <= Imag[i, j+1] and Imag[i, j] <= Imag[i, j-1]):
                    Iorig[i, j] = 0
    return Iorig

# I = cv2.cvtColor(cv2.imread('./images/museum.jpg'), cv2.COLOR_BGR2GRAY).astype(float)
# Iedge, Imag, Idir = findeges(I, 0.1, 10)
# Inms = nms(Iedge.copy(), Imag, Idir)
# plt.subplot(1,2,1)
# plt.imshow(Iedge, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(Inms, cmap='gray')
# plt.show()

# 2c)
def hysteresis(img, Imag, tlow, thigh):
    Imag[Imag < tlow] = 0
    highPixels = np.zeros((img.shape))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(I[i,j] >= thigh):
                highPixels[i,j] = 1

    comp, labels, stats, cen = cv2.connectedComponentsWithStats(Imag.astype(np.uint8), 20, 4, cv2.CV_16U)
    labels2 = labels.copy()
    labels2[highPixels == 1] = 1
    uLabels = np.unique(labels2)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] not in uLabels:
                labels[i,j] = 0

    img[labels > 0] = 1   
    return img

# I = cv2.cvtColor(cv2.imread('./images/museum.jpg'), cv2.COLOR_BGR2GRAY)
# # origI = I.copy()
# Iedge, Imag, Idir = findeges(I, 0.5, 20)
# # Inms = nms(Iedge.copy(), Imag, Idir)
# Inms = cv2.cvtColor(cv2.imread('./thinned1.png'), cv2.COLOR_BGR2GRAY)
# Ihis = hysteresis(Inms.copy(), Imag.copy(), 70, 100)
# Ihis[Ihis < 40] = 0
# Ihis[Ihis > 0] = 1
# plt.subplot(1,2,1)
# plt.imshow(Inms, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(Ihis, cmap='gray')
# plt.show()

### TASK 3 ###
# 3a)
# arr = np.zeros((300, 300))
# x = 10
# y = 10
# arr[x,y] = 1
# acc_arr = np.zeros((300,300))
# theta = np.deg2rad(np.linspace(-90, 90, 300))
# diagonal = np.ceil(np.sqrt((arr.shape[0])**2 + (arr.shape[1])**2))
# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         if(arr[i,j] <= 0):
#             continue
#         for k in range(theta.shape[0]):
#             ro = int(i*np.cos(theta[k]) + j*np.sin(theta[k]) + arr.shape[0]/2)
#             acc_arr[ro, k] += 1
# plt.imshow(acc_arr)
# plt.show()

# 3b)
def hough_find_lines(img, thetas, rhos, threshold, D):
    acc_arr = np.zeros((thetas.shape[0], rhos.shape[0]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] == 0):
                continue
            rho = i*np.cos(thetas) + j*np.sin(thetas)
            rho_bins = np.round(((rho + D)/(2*D)) * rho.shape[0]).astype(int)
            for theta in range(300):
                acc_arr[rho_bins[theta], theta] += 1
    return acc_arr
            
# I = cv2.imread('./images/oneline.png')
# # I = cv2.imread('./images/rectangle.png')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(float)
# D = np.ceil(np.sqrt(I.shape[0]**2 + I.shape[1]**2))
# I = findeges(I, 1, 5)[0]
# theta = np.deg2rad(np.linspace(-90, 90, 300))
# rho = np.linspace(-D, D, 300)
# plt.imshow(hough_find_lines(I, theta, rho, 50, D))
# plt.show()

# 3c)
def nonmaxima_suppression_box(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            left = max(0,i-1)
            right = max(0,i+1+1)
            bottom = max(0,j-1)
            top = max(0,j+1+1)
            neigh = img[left:right,bottom:top]
            for m in range(neigh.shape[0]):
                for n in range(neigh.shape[1]):
                    if(neigh[m,n] > img[i,j]):
                        img[i,j] = 0
    return img

# I = cv2.imread('./images/rectangle.png')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# D = np.ceil(np.sqrt(I.shape[0]**2 + I.shape[1]**2))
# I, Imag, Idir = findeges(I.copy(), 1, 25)
# theta = np.deg2rad(np.linspace(-90, 90, 300))
# rho = np.linspace(-D, D, 300)
# plt.imshow(nonmaxima_suppression_box(hough_find_lines(I.copy(), theta, rho, 50,D)))
# plt.show()

# 3 d)
# I = cv2.cvtColor(cv2.imread('./images/oneline.png'), cv2.COLOR_BGR2GRAY).astype(float)
# Iedge = findeges(I.copy(), 0.8, 100)[0]
# D = np.ceil(np.sqrt(I.shape[0]**2 + I.shape[1]**2))
# theta = np.deg2rad(np.linspace(-90, 90, 300))
# rho = np.linspace(-D, D, 300)
# max_rho = D
# hough_array = hough_find_lines(Iedge, theta, rho, 50, D)

# for i in range(hough_array.shape[0]):
#     for j in range(hough_array.shape[1]):
#         if(hough_array[i,j] > max_rho):
#             util.draw_line(rho[i], theta[j], max_rho)

# plt.imshow(I)
# plt.show()

# 3 e)
I = cv2.cvtColor(cv2.imread('./images/pier.jpg'), cv2.COLOR_BGR2GRAY).astype(float)
Iedge = findeges(I.copy(), 0.75, 20)[0]
D = np.ceil(np.sqrt(I.shape[0]**2 + I.shape[1]**2))
theta = np.deg2rad(np.linspace(-90, 90, 300))
rho = np.linspace(-D, D, 300)
max_rho = D
hough_array = nonmaxima_suppression_box(hough_find_lines(Iedge, theta, rho, 50, D))
lines = []
for i in range(hough_array.shape[0]):
    for j in range(hough_array.shape[1]):
        if(hough_array[i,j] > D):
            lines.append((i,j, hough_array[i,j]))

lines = sorted(lines,key=lambda x: x[2])
lines = lines[-10:len(lines)]
plt.axis([0, 640, 426, 0])
I = I.astype(np.uint8)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_GRAY2RGB))
for x,y,z in lines:
    util.draw_line(rho[x],theta[y],max_rho)
plt.show()