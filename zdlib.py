import numpy as np
import cv2

def neighbor_4(seed_point):
    r = [
        (seed_point[0] + 1, seed_point[1]), #down
        (seed_point[0] - 1, seed_point[1]), #up
        (seed_point[0], seed_point[1] + 1), #left
        (seed_point[0], seed_point[1] - 1)  #right
    ]
    return r

#Define floodfill separate output function
def FloodfillSeparate4(seed_point, in_image, out_img, new_color):
    height, width = in_image.shape #get height and width of input image
    frontier = [seed_point] #initialize frontier array with seed point
    old_color = in_image[seed_point[0],seed_point[1]] #get current color of the seed point
    if (old_color == new_color): # if the current color is the same with the new color, return image
        return

    out_img[seed_point[0],seed_point[1]] = new_color #set  color at seed point to new color

    #process each point in frontier array
    while frontier: # while frontier array is not empty
        q = frontier.pop(0) #get next point in array
        for r in neighbor_4(q): #check its 4 neighbors
            if 0 <= r[0] < height and 0 <= r[1] < width: #check if these neighbors are in bounds
                if in_image[r[0],r[1]] == old_color and out_img[r[0],r[1]]!=new_color:
                    # add to frontier the point where old color is present in input pictures and
                    # ouput is not yet floodfilled.
                    frontier.append(r)
                    out_img[r[0],r[1]] = new_color #fill pixel with new color value
    return out_img

def FloodfillSeparate8(seed_point, in_image, out_img, new_color):
    height, width = in_image.shape #get height and width of input image
    frontier = [seed_point] #initialize frontier array with seed point
    old_color = in_image[seed_point[0],seed_point[1]] #get current color of the seed point
    if (old_color == new_color): # if the current color is the same with the new color, return image
        return

    out_img[seed_point[0],seed_point[1]] = new_color #set  color at seed point to new color
    neighbors_8 = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    #process each point in frontier array
    while frontier: # while frontier array is not empty
        q = frontier.pop(0) #get next point in array
        for seedi, seedj in neighbors_8:  # Check its 8 neighbors
            r = (q[0] + seedi, q[1] + seedj)
            if 0 <= r[0] < height and 0 <= r[1] < width: #check if these neighbors are in bounds
                if in_image[r[0],r[1]] == old_color and out_img[r[0],r[1]]!=new_color:
                    # add to frontier the point where old color is present in input pictures and
                    # ouput is not yet floodfilled.
                    frontier.append(r)
                    out_img[r[0],r[1]] = new_color #fill pixel with new color value
    return out_img

def floodfill_f(seed_point, image, new_color):
    height, width = image.shape #get img dimension
    frontier = [seed_point] #initilize frontier array
    old_color = image[seed_point[0], seed_point[1]]

    if old_color == new_color:
        return

    while frontier:
        q = frontier.pop(0)
        image[seed_point[0], seed_point[1]] = new_color
        for r in neighbor_4(q):
            if 0 <= r[0] < height and 0 <= r[1] < width:  # Check bounds
                #if neighbors has old color, add to frontier
                if image[r[0], r[1]] == old_color:
                    frontier.append(r)
                    image[r[0], r[1]] = new_color # set new color value to pixel
    return image

def double_threshold(in_image,out_img,low_thresh,high_thresh):
    height, width = in_image.shape #get input img dimension
    high_thresh_img = np.zeros((height, width)) #create blank high threshold image
    low_thresh_img = np.zeros((height, width)) #create blank low threshold image
    for i in range(height):
        for j in range(width):
            # get high thresh image
            if in_image[i, j] >= high_thresh:
                high_thresh_img[i,j]=255
                #out_img[i, j] = 255
            else:
                high_thresh_img[i, j] = 0

            # get low thresh image
            if in_image[i, j] >= low_thresh:
                low_thresh_img[i,j]=100
                # set to 255 will go against floodfill algorithm as it return low thresh image as new color for
                # output image will be set to 255

            else:
                low_thresh_img[i,j]=0

    # iterate through the image coordinate as they all have same dimension
    for i in range(height):
        for j in range(width):
            # area where the high thresh image is set to 255, perform flood fill using
            # low thresh image as input and output on output image.
            if high_thresh_img[i,j]==255:
                FloodfillSeparate8((i, j), low_thresh_img, out_img, 255)

    #Show threshold images
    # cv2.imshow("High Threshold Image", high_thresh_img)
    # cv2.imshow("Low Threshold Image", low_thresh_img)
    return out_img

def get_list_pixexls(img):
    pixel_lists = []
    height, width = img.shape
    for i in range(256):
        pixel_lists.append([])

    for i in range(height):
        for j in range(width):
            g=int(img[(i, j)])
            pixel_lists[g].append((i,j))

    return pixel_lists

def histogram_equalization(in_img,out_img):
    height, width = in_img.shape #get img dimension
    total_pixel = height * width #calculate total pixel
    pixel_dict = {i: 0 for i in range(256)} #create a dictionary to store 256 pixel values
    #print(total_pixel)

    for i in range(0,height):
         for j in range(0,width):
            pixel_dict[in_img[i,j]]+=1
    #print(pixel_dict)

    # Calculate the cumulative distribution function
    cumulative_sum = 0
    cdf = {}
    for pixel_value in range(256):
        cumulative_sum += pixel_dict[pixel_value] # add the current pixel value's count to the cumulative sum
        cdf[pixel_value] = cumulative_sum / total_pixel #normalize the intensity by the total pixel
    #print(cdf)

    # equalized picture by mapping pixel value with cdf normalized value
    for i in range(0,height):
         for j in range(0,width):
            out_img[i,j] = int(255*cdf[in_img[i,j]])
    return out_img
def erosion4(in_image, output_img):
    height, width = in_image.shape  # Get image dimensions
    neighbors_4 = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Right, left, up, down

    for i in range(height):
        for j in range(width):
            seed_point = (i, j)  # Define seed point

            # Only check if the current pixel is 255
            if in_image[seed_point] == 255:
                is_eroded = True  # Assume erosion until a neighbor fails

                # Check all 4 neighbors
                for di, dj in neighbors_4:
                    ni, nj = i + di, j + dj  # Neighbor coordinates
                    # Check bounds and neighbor value
                    if not (0 <= ni < height and 0 <= nj < width and in_image[ni, nj] == 255):
                        is_eroded = False
                        break

                # Set output based on erosion result
                output_img[i, j] = 255 if is_eroded else 0
            else:
                output_img[i, j] = 0  # Set to 0 if the current pixel is not 255

    return output_img

# def erosion4(in_image,output_img):
#     height, width = in_image.shape #get img dimension
#     for i in range(height):
#         for j in range(width):
#             seed_point=(i,j) #define seed point
#
#             # if the value of the current pixel and its 4 neighbor are 255, turn the pixel at current pixel
#             # at ouput image to ON, otherwise, set it to 0.
#             if in_image[seed_point]==255:
#                 if (in_image[neighbor_4(seed_point)[0]] == 255 and #right
#                         in_image[neighbor_4(seed_point)[1]] == 255 and #left
#                         in_image[neighbor_4(seed_point)[2]] == 255 and #up
#                         in_image[neighbor_4(seed_point)[3]] == 255): #down
#                     #print("left",neighbor_4(seed_point)[0],"right",neighbor_4(seed_point)[1],"up",neighbor_4(seed_point)[2],"down",neighbor_4(seed_point)[3])
#                     output_img[i,j]=255
#                 else:
#                     #print("left",neighbor_4(seed_point)[0],"right",neighbor_4(seed_point)[1],"up",neighbor_4(seed_point)[2],"down",neighbor_4(seed_point)[3])
#                     output_img[i,j]=0
#     return output_img

def erosion8(in_image, output_img):
    height, width = in_image.shape  # Get image dimensions
    neighbors_8 = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connected neighbors

    for i in range(height):
        for j in range(width):
            seed_point = (i, j)  # Define seed point

            # Only check if the current pixel is 255
            if in_image[seed_point] == 255:
                # Check if all 8 neighbors are 255
                is_eroded = True
                for seedi, seedj in neighbors_8:
                    ni, nj = i + seedi, j + seedj
                    # Check bounds and neighbor value
                    if not (0 <= ni < height and 0 <= nj < width and in_image[ni, nj] == 255):
                        is_eroded = False
                        break

                output_img[i, j] = 255 if is_eroded else 0  # Set output based on erosion result
            else:
                output_img[i, j] = 0  # Set to 0 if the current pixel is not 255

    return output_img


def dilation4(in_image, out_img):
    height, width = in_image.shape #get img dimension

    #check each pixel in image, if the pixel in the input image is 255, set the output image at pixel to 255
    #check if neighbors of output image is in range, if they are also turn them to 255.
    for i in range(height):
        for j in range(width):

            if in_image[i, j] == 255:
                out_img[i, j] = 255
                for r in neighbor_4((i, j)):
                    if 0 <= r[0] < height and 0 <= r[1] < width:
                        out_img[r[0], r[1]] = 255
    return out_img


def dilation8(in_image, out_img):
    height, width = in_image.shape  # Get image dimensions
    neighbors_8 = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Check each pixel in the input image
    for i in range(height):
        for j in range(width):
            if in_image[i, j] == 255:  # If the current pixel is 255
                out_img[i, j] = 255  # Set the output pixel to 255
                # Turn all 8 neighbors to 255 if they are within the image bounds
                for seedi, seedj in neighbors_8:
                    ni, nj = i + seedi, j + seedj
                    if 0 <= ni < height and 0 <= nj < width:
                        out_img[ni, nj] = 255

    return out_img

def clean_image(in_image,out_image):
    height, width = in_image.shape
    temp_img = np.zeros((height, width), dtype=np.uint8)

    # Apply dilation first
    morphed_img  = dilation4(in_image, temp_img.copy())

    # Apply erosion multiple times
    for i in range(4):
        morphed_img  = erosion4(morphed_img , temp_img.copy())

    # Finally dilate to clean up, directly update the output image
    morphed_img = dilation4(morphed_img, temp_img.copy())

    out_image = dilation4(morphed_img , out_image)

    return out_image

# connected components

def connectedComponentRepeatedFloodFill(thresh_img,label_image):
    height,width = thresh_img.shape
    #label_image = np.zeros((height,width))
    label = 10  # Start label
    unique_l =[] # label values array
    num_components = 0

    for i in range(height):
        for j in range(width):
            if label_image[i, j] == -1 and thresh_img[i, j] == 255:  # ON state
                FloodfillSeparate4((i, j),thresh_img, label_image,  label)
                num_components += 1
                unique_l.append(label)
                label += 10  # Next label
    #cv2.imshow("ff:",label_image)

    return label_image, num_components,unique_l

#equivalent label value
def setequiv(equiv,img1,img2):
    equiv_img1 = getEquiv(equiv,img1)
    equiv_img2 = getEquiv(equiv,img2)

    if equiv_img1 > equiv_img2:
        equiv[equiv_img1]=equiv_img2
    else:
        equiv[equiv_img2]=equiv_img1

def getEquiv(equiv,label):
    if label not in equiv:
        equiv[label] = label  # Initialize if not present

    if equiv[label] != label:
        equiv[label] = getEquiv(equiv, equiv[label])  # Path compression

    return equiv[label]

def connected_components_union_find(thresh_img,proc_image):
    height, width = thresh_img.shape
    # Initialize intermediate processing arrays with bigger size for label more than 255
    #proc_image =np.zeros((height, width))
    # Initialize label output image
    label_image = np.zeros((height, width))
    label = 30  # Start labeling from 30
    equivalent_table = {} # To store equivalent labels

    for y in range(height-1):
        for x in range(width-1):
            seed = thresh_img[y,x]

            if seed ==255: #if not background
                if seed == thresh_img[y, x - 1] and seed == thresh_img[y - 1, x]: #check I(P)==I(L) and I(P)==I(U)
                    proc_image[y,x] = proc_image[y - 1,x] #set C(P)=C(U)
                    setequiv(equivalent_table,proc_image[y,x-1],proc_image[y - 1,x]) #Set U,L equivalent
                elif seed == thresh_img[y, x - 1]: #check I(P)==I(L)
                    proc_image[y,x] = proc_image[y,x-1] #set C(P)=C(L)
                elif seed == thresh_img[y - 1, x]: #check I(P)==I(U)
                    proc_image[y,x] = proc_image[y - 1, x] #set C(P)=C(U)
                else:
                    proc_image[y, x] = label #set C(P)=label
                    equivalent_table[label] = label #append label value to equivalent label
                    label += 1 #new label value

    #print("Equivalence table after first pass:", equivalent_table)

    unique_l=[] #unique label value set

    #second pass
    #iterate through image
    for y in range(height):
        for x in range(width):
            proc_image[y, x] = getEquiv(equivalent_table,proc_image[y,x]) #update image with its equivalent label value
            if proc_image[y,x]>0: #if not background
                if proc_image[y, x] not in unique_l: #append only unique label to array
                    unique_l.append(int(proc_image[y, x]))

    num_component = len(unique_l)

    #set processed cc-union find image from type int32 to type int8 as output label
    for label in unique_l:
        label_image[proc_image == label] = label
    #print("unique label",unique_l)
    #print("after 2nd pass",equivalent_table)
    return label_image,num_component,unique_l

#properties

def region_properties(label_img,label):
    m00 = m01 = m10 = m11 = m02 = m20 = 0
    u00 = u01 = u10 = u11 = u02 = u20 = 0
    height, width = label_img.shape

    #calculate moments
    for i in range(height):
        for j in range(width):
            if label_img[i,j]==label:
                m00 += 1 #zero order or area of the label object
                m01 += i #1st order (sums of y)
                m10 += j  # 1st order (sums of x)
                m11 += i*j #1st order x and y
                m02 += i**2 #2nd order (sums of y)
                m20 += j**2 #second order (sum of x)
                #print("m00:",m00,"m01",m01,"m10",m10)

    #calculate centroid
    xc,yc = (m10/m00,m01/m00)
    #print('xc:', xc,'yc:',yc)


    #calculate central moments
    for i in range(height):
        for j in range(width):
            if label_img[i,j]==label:

                u00 = m00 #zero order or area
                u01 += i-yc #1st order (sums of y-yc)
                u10 += j-xc  # 1st order (sums of x-xc)
                u11 += m11-yc*m10 #1st order x and y
                u02 += m02-yc*m01 #2nd order (sums of y-yc)^2
                u20 += m20-xc*m10 #second order (sum of x-xc)^2

    moments = [m00,m01,m10,m11,m02,m20] #store moments in an array
    central_moments = [u00,u01,u10,u11,u02,u20] #store central moments in another array
    area = m00

    return moments,central_moments,area

#PCA

def pca(moments,central_m):
    #moments = [m00, m01, m10, m11, m02, m20]
    #central_moments = [u00, u01, u10, u11, u02, u20]
    central_m = np.array(central_m, dtype=float)
    #eigen value 1
    ev1=1/(2*central_m[0])*(central_m[-1]+central_m[-2]+np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2))
    #eigen value 2
    ev2=1/(2*central_m[0])*(central_m[-1]+central_m[-2]-np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2))
    # angle
    theta = 0.5*np.atan2(2*central_m[-3],central_m[-1]-central_m[-2])
    # major axis length
    major_ax_length=2*np.sqrt(ev1)
    # minor axis length
    minor_ax_length = 2 * np.sqrt(ev2)
    #eccentricity
    eccentricity = np.sqrt(1-np.sqrt(ev2)/np.sqrt(ev1))
    # eccentricity = np.sqrt((2 * np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2)) /
    #                       (central_m[-1] + central_m[-2] + np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2)))
    return ev1,ev2,theta,major_ax_length,minor_ax_length,eccentricity

#### Object Detection ####
def turnright(direction):
    if direction == "N":
        direction = "E"
    elif direction == "E":
        direction = "S"
    elif direction == "S":
        direction ="W"
    else:
        direction = "N"
    return direction

def turnleft(direction):
    if direction == "N":
        direction = "W"
    elif direction == "E":
        direction = "N"
    elif direction == "S":
        direction ="E"
    else:
        direction = "S"
    return direction

def move(direction,seed,img): #seedpoint is (y,x)
    height,width = img.shape
    if seed[0]-1 >=0:
        if direction == "N":
            return tuple([seed[0]-1,seed[1]]) #move up (y-1)
    if seed[1]-1>=0:
        if direction =="W":
            return tuple([seed[0],seed[1]-1]) #move right (x-1)
    if seed[1]+1<width:
        if direction == "E":
            return tuple([seed[0],seed[1]+1]) #move right (x+1)
    if seed[0]+1<height:
        if direction == "S":
            return tuple([seed[0]+1,seed[1]]) #move down (y+1)


def isFrontOn(dir,in_img,seed,label):
    height,width = in_img.shape
    if dir == "N":
        if seed[0]-1>=0:
            if in_img[seed[0]-1,seed[1]] == label:
                return True

    elif dir == "W":
        if seed[1]-1 >=0:
            if in_img[seed[0],seed[1]-1] == label:
                return True
    elif dir == "E":
        if seed[1]+1  < width:
            if in_img[seed[0],seed[1]+1] == label:
                return True
    elif dir == "S":
        if seed[0] +1  < height:
            if in_img[seed[0]+1,seed[1]] == label:
                return True

    else:
        return False

def isLeftOn(dir,in_img,seed,label):
    height,width =in_img.shape

    if dir == "N":
        if seed[1] - 1 >= 0:
            if in_img[seed[0],seed[1]-1] == label: #check left of north
                return True
    elif dir == "E":
        if seed[0] - 1 >=0:
            if in_img[seed[0]-1,seed[1]] == label: # check left of east
                return True
    elif dir == "W":
        if seed[0] + 1 <height:
            if in_img[seed[0]+1,seed[1]] == label: #check left of west
                return True
    elif dir == "S":
        if seed[1] + 1 < width:
            if in_img[seed[0],seed[1]+1] == label: #check left of south
                return True


    else:
        return False

def wallfollowing(in_image,label):
    height,width = in_image.shape

    # initialize direction pointing north
    dir="N"
    found = False
    path = []

    # iterate and find first label pixel
    for i in range(height):
        if found == True:
            break
        for j in range(width):
            if in_image[i,j]==label:
                seed = (i,j)
                path.append(seed)
                found = True
                break
    current_s = seed

    #print("start:", current_s)
    #if front in label, keep turning right to get starting position
    while isFrontOn(dir,in_image,current_s,label):
        dir = turnright(dir)
        #print("Location A:",current_s,"direction:",dir)
    #turn right to start following wall
    dir = turnright(dir)
    #print("Location B:", current_s, "direction:", dir)
    #print(len(path))

    #repeat as long as starting seed is different from ending seed.
    while True:
        #print(len(path))
        if 0<=current_s[0]<height and 0<=current_s[1]<width:
            if isLeftOn(dir,in_image,current_s,label):
                dir = turnleft(dir)
                current_s = move(dir,current_s,in_image)
                path.append(current_s)
                #print("Location C:", current_s, "direction:", dir)

            elif not isFrontOn(dir,in_image,current_s,label):
                dir = turnright(dir)
                #print("Location D:", current_s, "direction:", dir)
            else:
                current_s = move(dir,current_s,in_image)
                path.append(current_s)
                #print("Location E:", current_s, "direction:", dir)
        if current_s == seed and len(path) > 1:
            break
    #print(path)

    return path

def Gaussian(sigma):
    ''' Create Gaussian kernel based on given sigma value'''
    a = round(2.5 * sigma - 0.5)
    w = 2 * a + 1
    G = np.zeros(w)
    sum = 0
    for i in range(w):
        G[i] = np.exp((-1*(i-a)*(i-a))/(2*sigma*sigma))
        sum +=G[i]
    G = G/sum
    return G,a

def Gaussian_Deriv(sigma):
    ''' Create Gaussian Derivative kernel based on given sigma value'''
    a = round(2.5 * sigma - 0.5)
    w = 2*a+1
    Gderiv = np.zeros(w)
    sum = 0

    for i in range(w):
        Gderiv[i]=-1*(i-a)*np.exp((-1*(i-a)*(i-a))/(2*sigma*sigma))
        sum += -i * Gderiv[i]
    Gderiv=Gderiv/sum
    return Gderiv

def Convolve(in_img,kernel,out_img):
    ''' This convolution function convolve an image 2D with a kernel 2D'''
    height, width = in_img.shape
    k_h,k_w = kernel.shape

    center = (k_h//2,k_w//2)
    padded_img = np.pad(in_img.copy(), ((center[0], center[0]), (center[1], center[1])), mode='constant')

    for i in range(height):
        for j in range(width):
            sum = 0
            for k in range(k_h):
                for m in range(k_w):
                    offseti = -1*center[0]+k
                    offsetj = -1*center[1]+m
                    if i+offseti in range(height):
                        if j+offsetj in range (width):
                            sum += padded_img[i + center[0] + offseti, j + center[1] + offsetj]*kernel[k,m]
            out_img[(i,j)]=sum
    return out_img

def ConvolveSeparable(in_img, gx, gy, half_w):

    '''
    This function convolve a 2D array with an 1D separably vertically and horizontally for
    faster run time than convolution itself.
    The reason why iteration is starting at half_width and end at width-half_w is because
    by using half_width at starting, there's no need to flip the kernel.
    '''

    height, width = in_img.shape #get image shape
    tmp = np.zeros((height, width))
    output = np.zeros((height, width))

    # Step 1: Horizontal Convolution
    # the idea is to convolve horizontally before convolve vertically, in this case, parameter gx meant
    # horizontal convolution with gy
    for y in range(height):
        for x in range(half_w,width-half_w):
            sum = 0
            for i in range(len(gx)):
                if x + half_w-i < width:  # Ensure stay within bounds
                    sum += gx[i] * in_img[y, x +half_w- i]
            tmp[y, x] = sum

    # Step 2: Vertical Convolution
    # the idea is to convolve horizontally before convolve vertically, in this case, parameter gy meant
    # vertical convolution with gx
    for y in range(half_w,height-half_w):
        for x in range(width):
            val = 0
            for i in range(len(gy)):
                if y + half_w - i < height:  # Ensure stay within bounds
                    val += gy[i] * tmp[y + half_w - i, x]
            output[y, x] = val

    return output

def MagnitudeGradient(gx,gy):
    '''Calculate magnitude and gradient of the image based on horizontal and vertical gradient'''
    # use np instead of looping for faster computation as numpy perform operation element-wise by default
    magnitude= np.sqrt(gy ** 2 + gx ** 2)

    #similar to angle calculation, numpy perform element wise computation
    angle = np.arctan2(gy,gx)

    return magnitude,angle


def NonMaxSuppression(G,theta):
    '''
    This function take in G as magnitude image and theta as gradient image
    '''
    height, width = G.shape
    sup_img = G.copy()

    #convert theta to degrees and only takes in range of 0 to 180
    theta = (theta + np.pi) % np.pi*180/np.pi

    for y in range (height):
        for x in range(width):
            if theta[y,x] <= 22.5 or theta[y,x] > 157.5:
                if x - 1 >= 0 and x + 1 < width: #check bounds
                    #check local min left right
                    if G[y, x] < G[y, x - 1] or G[y, x] < G[y, x + 1]:
                        sup_img[y,x]=0
            elif 22.5 < theta[y,x] <= 67.5:
                if y-1 >=0 and x-1 >= 0 and y+1 < height and x+1 <width: #check bounds
                    #check local min top-left, bottom-right
                    if G[y,x]< G[y-1,x-1] or G[y,x]< G[y+1,x+1]:
                        sup_img[y,x]=0
            elif 67.5 < theta[y,x] <= 112.5:
                if y - 1 >= 0 and y + 1 < height:#check bounds
                    #check local min top , bottom
                    if G[y, x] < G[y - 1, x] or G[y, x] < G[y + 1, x]:
                        sup_img[y,x]=0
            else:
                if y-1 >=0 and x+1 < width and y+1 < height and x-1 >= 0:#check bounds
                    #check local min bottom-left,top-right
                    if G[y, x] < G[y - 1, x + 1] or G[y, x] < G[y + 1, x - 1]:
                        sup_img[y,x]=0
    return sup_img #return suppression image

def Hysteresis(sup):

    ''' This function perform hysteresis by setting threshold value and perform edges linking based on hysteresis.'''
    height, width = sup.shape
    hyst = sup.copy()

    sup_list = np.sort(sup.flatten()) #flatten the suppression image into 1D array to sort
    #print(sup_list)
    t_high = np.percentile(sup_list,90) #high threshold is at 90% of sorted array
    #print(t_high)
    if t_high == 0:
        t_high = np.mean(sup_list[sup_list > 0])
    #print(t_high)
    t_low = 0.2*t_high #low threshold is 20% of high threshold value
    #print(t_low)

    for y in range(height):
        for x in range(width):
            if hyst[y,x] >= t_high: #strong edges
                hyst[y,x] = 255
            elif hyst[y,x] >= t_low: #weak edges
                hyst[y,x] = 125
            else:
                hyst[y,x] = 0  #not edges

    edges = hyst.copy()
    for y in range(height):
        for x in range(width):
            if hyst[y,x] == 125: #if weak edges
                if y-1 >= 0 and x-1 >=0 and y+1 < height and x+1 < width: #check bounds
                    #if any 8 neighbors is strong edges, turn pixel at [y,x] to strong edges, else, suppress it.
                    if (hyst[y - 1, x - 1] == 255 or hyst[y - 1, x] == 255 or hyst[y - 1, x + 1] == 255 or
                            hyst[y, x - 1] == 255 or hyst[y, x + 1] == 255 or
                            hyst[y + 1, x - 1] == 255 or hyst[y + 1, x] == 255 or hyst[y + 1, x + 1] == 255):
                        edges[y,x] = 255
                    else:
                        edges[y,x] = 0
    return edges

def manhattan_chamfer_distance(img):
    ''' This function calculate the Manhattan Chamfer Distance of an image to its edges.'''
    height, width = img.shape
    out = np.zeros((height, width))
    #1st pass, moving top to bottom, left to right
    for y in range(height):
        for x in range(width):
            if img[y,x] == 255: #turn edge pixels to 0 as they are 0 distance from themselves
                out[y,x] =0
                # print("here1a")
            else:
                if x -1 <0:#check bounds
                    left=np.inf
                else:
                    left = out[y,x-1]
                if y-1 <0:
                    top = np.inf
                else:
                    top = out[y-1,x]
                    # take minium of
                out[y,x] = min(np.inf,1 + left,1 + top)
                # print("here1b")
    #2nd pass, moving bottom to top, right to left
    for y in range(height-1,-1,-1): #traverse at last index to by step size of 1
        for x in range(width-1,-1,-1):
            if img[y,x] != 255:
                if x+1 >= width:
                    right = np.inf
                else:
                    right = out[y,x+1]
                if y+1 >= height:
                    bot = np.inf
                else:
                    bot = out[y+1,x]
                out[y,x] = min(out[y,x],1+right,1+bot)
                # print("here2")
    return out


def ridler_calvard_threshold(image):
    # Convert image to grayscale if it's not already

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initial threshold

    T = 100

    T_prev = 0

    max_iterations = 100

    epsilon = 1e-5

    for _ in range(max_iterations):

        # Calculate the foreground and background means

        foreground = image[image >= T]

        background = image[image < T]

        if len(foreground) == 0:

            mean_foreground = 0

        else:

            mean_foreground = np.mean(foreground)

        if len(background) == 0:

            mean_background = 0

        else:

            mean_background = np.mean(background)

        # Update threshold

        T = (mean_foreground + mean_background) / 2

        # Check for convergence

        if abs(T - T_prev) < epsilon:
            break

        T_prev = T

    return T

def template_matching(template, original_img):

    ''' This function calculate template matching by calculating sum of squares difference'''
    height, width = original_img.shape
    h, w = template.shape

    # Centre of the template
    cy = h // 2 
    cx = w // 2

    best_position = (0, 0)
    min_ssd = np.inf
    # scaling factor as ssd and differences tend to be very big based on image size
    #scale = 0.000000000000000000000000000000001

    #traverse from mid point of template in respect to 0,0 to mid point of template in respect to size-midpoint for less
    #number of iterations than starting at 0,0. This helps with run time
    for i in range(cy, height - cy):
        for j in range(cx, width -cx):
            ssd = 0
            # Template elements
            # iterate from beginning to end of the template with respect to center point
            for y in range(-cy, cy):
                for x in range(-cx, cx):

                    # Calculating differences based on specific index
                    if 0<=i+y<height and 0<=j+x<width:
                        #print("I:",(i + y, j + x),"T:",(cy + y, cx + x))
                        #log scale the number so it doesnt jump to infinity after scaling
                        diff = np.log1p((np.float64(original_img[i + y, j + x]) - np.float64(template[cy + y, cx + x]))**2)
                        # if err == np.inf:
                        #     print("error:", err, (y, x))
                        #     break
                        # if err<50:
                        #     ssd+= err
                        # elif 50<= err < np.inf: #scale down when too big
                        #     ssd+= err*scale
                        if diff<np.inf:
                            ssd+=diff
            #print("ssd:",ssd,(i,j))
            if min_ssd > ssd:
                #updating minimum ssd, if 2 images have smallest ssd at same point, this is likely that
                #that is where they matched it other
                min_ssd = ssd
                best_position = (i, j)
                # print("best position", best_position)
                # print("min", min_ssd)
    return best_position

def get_border(template, best_position):
    ''' this function calculate the border from the given position. A note here is that the position calculated from
        template_matching function is the center pixel as we traverse starting at midpoint of template, therefore, the
        border of the template is returned as an array with respect to its center and not top left point.
    '''
    template_height, template_width = template.shape
    border_indices = []

    # Calculate the offsets for border rows and columns
    top = best_position[0] - template_height // 2
    bottom = best_position[0] + template_height // 2
    left = best_position[1] - template_width // 2
    right = best_position[1] + template_width // 2

    # Add top and bottom rows
    for x in range(left, right + 1):
        border_indices.append((top, x))
        border_indices.append((bottom, x))

    # Add left and right columns (excluding corners already added)
    for y in range(top + 1, bottom):
        border_indices.append((y, left))
        border_indices.append((y, right))

    return border_indices

def watershed(quant_mag):
    img = quant_mag.copy()
    img = img.astype(np.uint8)
    height,width = img.shape
    label_img = np.zeros((height,width))
    label = -1
    label_img[:,:]=label

    frontier=[]
    globallabel=1
    neighbor_8 = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    p_lists = get_list_pixexls(img.astype(np.uint8))
    #print(p_lists)
    #for g in np.unique(img):
    for g in range(256):
        #(a) create frontier
        temp_label = label_img.copy()

        for (i, j) in p_lists[g]:
            # Check if any neighboring pixel has a label >= 0 in label_img
            for seedi, seedj in (neighbor_8):
                q = (i +seedi, j + seedj)
                if 0 <= q[0] < height and 0 <= q[1] < width:
                    if temp_label[q]>=0:
                        label_img[i,j]=label_img[q]
                        frontier.append(q)
                        #print("a: label img at p",label_img[i,j])

        #(b) grow basins
        while frontier:
            p = frontier.pop(0)
            i,j = p
            for seedi, seedj in (neighbor_8):
                q = (i + seedi, j + seedj)
                if 0 <= q[0] < height and 0 <= q[1] < width:
                    if img[q]==g and label_img[q]==-1:
                        label_img[q]=label_img[p]
                        frontier.append(q)
                        #print("b: label img at q",label_img[q])
        #(c) create new catchment basins
        for (i,j) in p_lists[g]:
            if label_img[i,j]==-1:
                globallabel+=1
                #print("c:",globallabel)
                FloodfillSeparate8((i,j),img,label_img,globallabel)

    return label_img


def markerwatershed(quant_mag, marker):
    img = quant_mag.copy()
    img = img.astype(np.uint8)
    height, width = img.shape

    # Pad the image and label image with -1 for label and 0 for image to prevent overflow
    padded_img = np.pad(img, pad_width=1, mode='constant', constant_values=0)  # 0 for background
    # padded_label_img = np.pad(np.zeros((height, width), dtype=int), pad_width=1, mode='constant', constant_values=-1)

    # Create a list to track the frontier during watershed
    frontier = []

    label_img = np.zeros((height,width))
    label = -1
    label_img[:,:]=label

    # Start with a label
    globallabel = 1
    neighbor_8 = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Get the list of pixels (assuming get_list_pixexls is implemented)
    p_lists = get_list_pixexls(padded_img)  # Get the list of pixels for processing

    # (c) create new catchment basins
    label_img, num_components, unique_l = connectedComponentRepeatedFloodFill(marker, label_img)

    #padded in order to enclose the connected component
    padded_label_img = np.pad(label_img, pad_width=1, mode='constant', constant_values=-1)

    # Display the labeled image
    label_img_imshow = label_img.astype(np.uint8)
    cv2.imshow("CCFF", label_img_imshow)
    # print(num_components)
    # print(unique_l)

    # Iterate over intensity levels (0-255)
    for g in range(256):
        temp_label = padded_label_img.copy()  # Make a copy for label propagation
        # (a) create initial frontier
        for (i, j) in p_lists[g]:
            # Check if any neighboring pixel has a label >= 0 in label_img
            for seedi, seedj in neighbor_8:
                q = (i + seedi, j + seedj)
                if 0 <= q[0] < height and 0 <= q[1] < width:
                    if temp_label[q] >= 0:
                        padded_label_img[i, j] = padded_label_img[q]
                        frontier.append((i, j))

        # (b) Grow basins
        while frontier:
            p = frontier.pop(0)
            i, j = p
            for seedi, seedj in neighbor_8:
                q = (i + seedi, j + seedj)
                if 0 <= q[0] < height and 0 <= q[1] < width:
                    if padded_img[q] <= g and padded_label_img[q] == -1:
                        padded_label_img[q] = padded_label_img[p]
                        frontier.append(q)


    # Crop the padded label image to remove the padding
    final_label_img = padded_label_img[1:-1, 1:-1]

    return final_label_img, num_components, unique_l

# def markerwatershed(quant_mag,marker):
#     img = quant_mag.copy()
#     img=img.astype(np.uint8)
#     height,width = img.shape
#     label_img = np.zeros((height,width))
#     label = -1
#     label_img[:,:]=label
#
#     frontier=[]
#     globallabel= 1
#     neighbor_8 = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
#     p_lists = get_list_pixexls(img)
#     # (c)
#
#     label_img, num_components, unique_l = connectedComponentRepeatedFloodFill(marker, label_img)
#     label_img_imshow = label_img.astype(np.uint8)
#     cv2.imshow("CCFF",label_img_imshow)
#     print(num_components)
#     print(unique_l)
#     #print(p_lists)
#
#     #for g in np.unique(img):
#     for g in range(256):
#         temp_label = label_img.copy()
#         #(a)
#         for (i, j) in p_lists[g]:
#             # Check if any neighboring pixel has a label >= 0 in label_img
#             for seedi, seedj in (neighbor_8):
#                 q = (i +seedi, j + seedj)
#                 if 0 <= q[0] < height and 0 <= q[1] < width:
#                     if temp_label[q]>=0:
#                         label_img[i,j]=label_img[q]
#                         frontier.append((i,j))
#
#                         #print("a: label img at p",label_img[i,j])
#         #(b)
#         while frontier:
#             p = frontier.pop(0)
#             i,j = p
#             for seedi, seedj in (neighbor_8):
#                 q = (i + seedi, j + seedj)
#                 if 0 <= q[0] < height and 0 <= q[1] < width:
#                     if img[q]<=g and label_img[q]==-1:
#                         label_img[q]=label_img[p]
#                         frontier.append(q)
#                         #print("b: label img at q",label_img[q])
#
#     return label_img,num_components,unique_l