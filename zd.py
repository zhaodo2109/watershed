from zdlib import *
from tkinter import Tk
import cv2
import numpy as np
from tkinter import filedialog, simpledialog

import time

print("OpenCV Version:", cv2.__version__)
print("Numpy Version: ", np.__version__)

Tk().withdraw()
filename = filedialog.askopenfilename(title="Select an Image") # open img file

image = cv2.imread(filename, 0)
color_image = cv2.imread(filename, 1)
out_img = color_image.copy()
cv2.imshow("Original Image", color_image) #show original image

height, width = image.shape #get img dimension
height1, width1,channels = color_image.shape #get img dimension

sigma = simpledialog.askfloat("Gaussian Sigma", "Enter the sigma value:")
start_time = time.time()

g,half_w = Gaussian(sigma)
g_der = Gaussian_Deriv(sigma)

# Horizontal Gradient by smoothing with Gaussian vertically first then convolve the result with
# Gaussian Derivative kernel
gx=ConvolveSeparable(image,g_der,g,half_w)
# cv2.imshow("Horizontal Gradient", gx)

# Vertical Gradient by smoothing with Gaussian horizontally first then convolve the result with
# Gaussian Derivative kernel
gy=ConvolveSeparable(image,g,g_der,half_w) #Vertical Gradient
# cv2.imshow("Vertical Gradient", gy)

G,angle = MagnitudeGradient(gx,gy)
G_histogram_equalization = np.zeros((height,width))
#cv2.imshow("Normal Watershed: Magnitude", G.astype(np.uint8))
G_histogram_equalization=histogram_equalization(G.astype(np.uint8),G_histogram_equalization)
cv2.imshow("Normal Watershed: Histogram Equalized Magnitude", G_histogram_equalization.astype(np.uint8))
#cv2.imshow("Gradient Image",angle)


'''Normal watershed'''
watershed_img = watershed(G)
max_label = np.max(watershed_img)
Lshow = (255 * watershed_img) / max_label if max_label > 0 else watershed_img

cv2.imshow("Normal Watershed: Labels",Lshow.astype(np.uint8))

'''Marker watershed'''
#cv2.imshow("Marker Watershed: Magnitude", G.astype(np.uint8))
G_histogram_equalization=histogram_equalization(G.astype(np.uint8),G_histogram_equalization)
#cv2.imshow("Marker Watershed: Histogram Equalized Magnitude", G_histogram_equalization.astype(np.uint8))

marker_thresh = np.zeros((height,width))
flipped_thresh = np.zeros((height,width))


t = ridler_calvard_threshold(image)
#print(t)
for i in range(height):
    for j in range(width):
        if image[i,j]>=t:
            marker_thresh[i,j]=255
        else:
            marker_thresh[i,j]=0

#double thresholding using threshold values from ridler calvard
marker_thresh=double_threshold(image,marker_thresh,t/1.5,t)


#flip the threshold image as in this case, background is 255
for i in range (height):
    for j in range(width):
        if marker_thresh[i,j]==0:
            flipped_thresh[i,j]=255
        else:
            flipped_thresh[i,j]=0

cv2.imshow("Marker Watershed: Threshold", flipped_thresh)

#chamfer distnce
chamfer_img=manhattan_chamfer_distance(flipped_thresh)


#normalize chamfer
min_val = np.min(chamfer_img)
max_val = np.max(chamfer_img)

scaled_chamfer = (chamfer_img - min_val) / (max_val - min_val) * 255

cv2.imshow("Marker Watershed: Chamfer", scaled_chamfer.astype(np.uint8))

chamfer_watershed= watershed(scaled_chamfer)
max_label = np.max(chamfer_watershed)
Lshow2 = (255 * chamfer_watershed) / max_label if max_label > 0 else chamfer_watershed

cv2.imshow("Marker Watershed: Watershed of Chamfer",Lshow2.astype(np.uint8))

''' '''
# Horizontal Gradient by smoothing with Gaussian vertically first then convolve the result with
# Gaussian Derivative kernel
gx=ConvolveSeparable(chamfer_watershed,g_der,g,half_w)
cv2.imshow("Horizontal Gradient", gx)

# Vertical Gradient by smoothing with Gaussian horizontally first then convolve the result with
# Gaussian Derivative kernel
gy=ConvolveSeparable(chamfer_watershed,g,g_der,half_w) #Vertical Gradient
cv2.imshow("Vertical Gradient", gy)

chamfer_w_mag,angle = MagnitudeGradient(gx,gy)

chamfer_w_mag_scale = 200*chamfer_w_mag
cv2.imshow("Magnitude Gradient", chamfer_w_mag_scale)

cv2.imshow("Gradient Image",angle)

# Suppression
suppression_wat_chamf = NonMaxSuppression(chamfer_w_mag_scale.astype(np.uint8),angle)
#cv2.imshow("Suppressed Image", suppression_wat_chamf)


#turn all pixels to on if >0 from magnitude
for i in range(height):
    for j in range(width):
        if suppression_wat_chamf[i,j]>0:
            suppression_wat_chamf[i,j]=255

# edgelinking
edges_watered_chamfer = Hysteresis(suppression_wat_chamf)
#cv2.imshow('edgy',edges_watered_chamfer)

#preprocessing image
test1 = np.zeros((height,width))
test2 = test1.copy()
test3 = test1.copy()
test4 = test1.copy()


dilation8(edges_watered_chamfer,test1)
#dilation8(test1,test2)

#erosion8(test2,test3)
erosion4(test1,test4)
# dilation8(test1,test2)
# dilation8(test2,test3)



cv2.imshow("Marker Watershed: Edges separating objects",test4)


# dilation(suppression_wat_chamf,suppression_wat_chamf)
#bitwise or operation


# Bitwise OR operation
marker_img = test4.astype(np.uint8) | flipped_thresh.astype(np.uint8)

cv2.imshow("Marker Watershed: Watershed Marker",marker_img)


#marker wathershed
marker_w_img,num_obj,labels = markerwatershed(G,marker_img)
cv2.imshow("Marker Watershed: Label Image",marker_w_img.astype(np.uint8))

# erode = np.zeros((height,width))
# erode = erosion8(marker_w_img,erode)
for obj in range(num_obj):
    label = labels[obj]
    #moments,central_m,area = region_properties(marker_w_img,label)
    path = wallfollowing(marker_w_img, label)
    #print(path)
    for pixels in path:
        out_img[pixels] = (0,128,0)
cv2.imshow("Marker Watershed: Label Image Highlight",out_img)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows() # close all image windows
