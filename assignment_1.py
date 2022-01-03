import sys
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
# 1a)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/umbrellas.jpg')
# plt.imshow(I)
# plt.show()

# 1b)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/umbrellas.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# I = I.astype(float)
# cRed = I[:, :, 0]
# cGreen = I[:, :, 1]
# cBlue = I[:, :, 2]
# matResult = (cRed + cGreen + cBlue)/3
# matResult = matResult.astype(np.uint8)
# plt.imshow(matResult)
# plt.show()

# 1c)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/umbrellas.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# I = I.astype(float)
# cRed = I[130:260, 240:450, 0]
# cRed = cRed.astype(np.uint8)
# plt.imshow(cRed, cmap='gray')
# plt.show()

# 1d)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/umbrellas.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# matResult = (255-I[130:260,240:450])
# I[130:260, 240:450] = matResult
# plt.imshow(I, cmap='gray')
# plt.show()

# 1e)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/umbrellas.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# img = np.copy(I)
# img = img*0.25
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(I)
# plt.show()

###### 2. TASK ######
# 2a)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
# old = np.copy(I)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# I[I<60] = 0
# I[I>=60] = 1
# plt.subplot(1, 2, 1)
# plt.imshow(old)
# plt.subplot(1,2,2)
# plt.imshow(I, cmap='gray')
# plt.show()

### using np.where() ###
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
# originalImage = np.copy(I)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# I = np.where(I<60, 0, 1)
# plt.subplot(1, 2, 1)
# plt.imshow(originalImage)
# plt.subplot(1,2,2)
# plt.imshow(I, cmap='gray')
# plt.show()

# 2b)
# def myhist(img, n):
#     I = img.reshape(-1)
#     H = np.zeros(n)
#     limit = round(255/n)
#     for i in range(I.shape[0]):
#         bin = int(I[i]/limit)
#         H[bin-1] = H[bin-1] + 1
#     H = H/np.sum(H)
#     plt.bar(np.arange(n), H)
#     plt.show()
# img = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# myhist(img, 20)

# 2c)
# def myhist(img, n):
#   I = img.reshape(-1)
#   min_v = min(I)
#   max_v = max(I)
#   H = np.zeros(n)
#   limit = round((max_v-min_v)/n)
#   for i in range(I.shape[0]):
#       bin = int(I[i]/limit)
#       H[bin-1] = H[bin-1] + 1
#   H = H/np.sum(H)
#   plt.bar(np.arange(n), H)
#   plt.show()
# img = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# myhist(img, 20)

# 2d) 
# def myhist(img, n):
#   I = img.reshape(-1)
#   H = np.zeros(n)
#   limit = round(255/n)
#   for i in range(I.shape[0]):
#       bin = int(I[i]/limit)
#       H[bin-1] = H[bin-1] + 1
#   H = H/np.sum(H)
#   return H
# n = 20
# LLight = cv2.imread('C:/Users/Uporabnik/Downloads/LLight.jpg')
# MLight = cv2.imread('C:/Users/Uporabnik/Downloads/MLight.jpg')
# HLight = cv2.imread('C:/Users/Uporabnik/Downloads/HLight.jpg')

# LLight = cv2.cvtColor(LLight, cv2.COLOR_BGR2GRAY)
# MLight = cv2.cvtColor(MLight, cv2.COLOR_BGR2GRAY)
# HLight = cv2.cvtColor(HLight, cv2.COLOR_BGR2GRAY)

# plt.subplot(1, 3, 1)
# plt.bar(np.arange(n), myhist(LLight, n))
# plt.subplot(1, 3, 2)
# plt.bar(np.arange(n), myhist(MLight, n))
# plt.subplot(1, 3, 3)
# plt.bar(np.arange(n), myhist(HLight, n))
# plt.show()

# Interpretation: the image with the lowest volume of light has peak at the very beginning of histogram
# which makes sense as there were many pixels that are almost black and have low values. It is also visible that the distribution 
# isn't really equal as there are a lot of "dark" pixels and not so much "light" ones.
# The image with middle volume of light has the most equal histogram as there are many different values but none of them is extremely high/low.
# The image with highest volume of light has very high peak somewhere in the middle which tells us that there are a lot of pixels
# being shined on by light

# 2e)
def otsu(gray, count_pixel, n):
    weight = 1/count_pixel
    his, bins = np.histogram(gray, np.arange(n))
    threshold, t_value, mi_bck, mi_fg = 0, 0, 0, 0
    bin_no = np.arange(n-1)
    for i in bins:
        bck = np.sum(his[:i]) * weight
        fg = np.sum(his[i:]) * weight
        if(bck != 0):
            mi_bck = np.sum(bin_no[:i]*his[:i]) / bck
        if(fg != 0):
            mi_fg = np.sum(bin_no[i:]*his[i:]) / fg
        value = bck * fg * (mi_bck - mi_fg) ** 2
        if value > t_value:
            t_value = value
            threshold = i
    return threshold

I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
pixel_c = I.shape[0]*I.shape[1]
value = otsu(I, pixel_c, 256)
I = np.where(I<value, 0, 1)
plt.imshow(I, cmap='gray')
plt.show()

##### 3. TASK #####
#3a)
# n = 5
# SE = np.ones((n,n), np.uint8)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/mask.png')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# I[I<55] = 0
# I[I>=55] = 1
# I_dil = np.copy(I)
# ### opening
# I_opening = cv2.erode(I, SE)
# I_opening = cv2.dilate(I_opening, SE)
# ### closing
# n = 8
# SE = np.ones((n,n), np.uint8)
# I_closing = cv2.dilate(I_dil, SE)
# I_closing = cv2.erode(I_closing, SE)
# plt.subplot(1,3,1)
# plt.imshow(I)
# plt.subplot(1,3,2)
# plt.imshow(I_opening, cmap='gray')
# plt.subplot(1,3,3)
# plt.imshow(I_closing, cmap='gray')
# plt.show()

# 3b)
# n = 25
# SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# I[I<55] = 0
# I[I>=55] = 1
# I = cv2.dilate(I, SE)
# I = cv2.erode(I, SE)
# plt.imshow(I, cmap='gray')
# plt.show()

# 3c)
def immask(img, mask):
    img[mask == 0] = 0
    return img

# n = 25
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/bird.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# mask = np.copy(I)
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# mask[mask<55] = 0
# mask[mask>=55] = 1
# plt.imshow(immask(I, mask))
# plt.show()

# 3d)
# n = 25
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/eagle.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# mask = np.copy(I)
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# threshold = otsu(mask, I.shape[0]*I.shape[1], 256)
# mask = np.where(mask<threshold, 1, 0)
# plt.imshow(immask(I, mask))
# plt.show()

# 3e)
# I = cv2.imread('C:/Users/Uporabnik/Documents/3. letnik/UZ/assignment_1/images/coins.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# mask = np.copy(I)
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# mask[mask<220] = 1
# mask[mask>=220] = 0
# comp, labels, stats, cen = cv2.connectedComponentsWithStats(mask, 220, 4, cv2.CV_16U)
# sizes = stats[1:, -1]
# for i in range(len(sizes)):
#     if sizes[i] > 700:
#         mask[labels == i+1] = 0
# I = immask(I, mask)
# plt.imshow(I)
# plt.show()