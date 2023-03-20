# Auxiliary functions for the Environment
import math
import numpy as np
import cv2
import skimage.exposure

from math import sin, cos
from numpy.random import default_rng

def ConvertToColorMap(img):
    img = np.copy(img)
    img = np.uint8(255*cv2.merge([img,img,img]))
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = img[:,:,::-1]

    return img

def ConvertToColor(img):
    img = np.copy(img)
    img = np.uint8(255*cv2.merge([img,img,img]))

    return img


def downsampling(image,r,c):

   # Blur with Gaussian kernel of width sigma=1
    img = cv2.GaussianBlur(image, (0, 0), 1, 1)

    # Downsample
    img = cv2.resize(img, (0, 0), fy=r, fx=c,
                           interpolation=cv2.INTER_AREA)

    return img


def upsampling(image,r,c):

    # Upsample
    img = cv2.resize(image, (0, 0), fy=r, fx=c,
                           interpolation=cv2.INTER_CUBIC)

    return img


def tan(theta):
    return np.tan((np.pi/180)*theta)


def generate_unc_map_blobs(h_field,w_field, beta_decay):

	#rng = default_rng(seed=seedval)
	rng = default_rng()

	# create random noise image
	noise = rng.integers(0, 255, (h_field,w_field), np.uint8, True)

	# blur the noise image to control the size
	blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

	# stretch the blurred image to full dynamic range
	stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

	# threshold stretched image to control the size
	thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

	# apply morphology open and close to smooth out and make 3 channels
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
	mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	# show results
	return np.where(mask==255,1,beta_decay)


def generate_unc_map_lines(h_field,w_field, beta_decay): # self

    theta_dg = np.random.randint(0,361,1)[0] # 90
    #print(theta_dg)
    theta_b = np.arctan(h_field/w_field)*(180/np.pi)
    theta_rad = (np.pi/180)*(theta_dg)

    #print("theta = {}°".format(theta_dg))

    mask = np.zeros((h_field,w_field), np.uint8)
    img_lines = np.zeros((h_field,w_field))
    img_unc_lines = np.zeros((h_field,w_field))

    radius = np.max((w_field,h_field)) # gambiarra, the radius must fill all the pixels in a line
    r, c = round(img_lines.shape[0]/2), round(img_lines.shape[1]/2)  # center row, center column

    half_line_1 = (int(c+radius*cos(theta_rad)), int(r+radius*sin(theta_rad)))
    half_line_2 = (int(c-radius*cos(theta_rad)), int(r-radius*sin(theta_rad)))

    cv2.line(img_lines, (c, r), half_line_1, color=255)
    cv2.line(img_lines, (c, r), half_line_2, color=255)

    line_coord = np.where(img_lines)
    ac, bc = line_coord[0], line_coord[1] #np.argmin(line_coord, axis=1)

    if 0 <= theta_dg <= theta_b:
        c1,c2,c3,c4 = [bc[0],ac[0]],[0,h_field],[w_field,h_field],[w_field,ac[-1]]

    elif theta_b < theta_dg <= 180-theta_b:
        c1,c2,c3,c4 = [0,ac[0]],[0,h_field],[bc[-1],h_field],[bc[0],ac[0]]

    elif 180-theta_b < theta_dg <= 180:
        c1,c2,c3,c4 = [0,0],[0,ac[-1]],[w_field,ac[0]],[w_field,0]

    elif 180 < theta_dg <= 180+theta_b:
        c1,c2,c3,c4 = [0,0],[0,ac[0]],[w_field,ac[-1]],[w_field,0]

    elif 180+theta_b < theta_dg <= 360-theta_b:
        c1,c2,c3,c4 = [bc[0],ac[0]],[bc[-1],h_field],[w_field,h_field],[w_field,0]

    elif 360-theta_b < theta_dg <= 360:
        c1,c2,c3,c4 = [0,ac[-1]],[0,h_field],[w_field,h_field],[w_field,ac[0]]

    pts = np.array([c1,c2,c3,c4])
    _=cv2.drawContours(mask, np.int32([pts]),0, 255, -1)

    img_unc_lines[mask==0] = beta_decay
    img_unc_lines[mask>0] = 0.95 # 1

    return img_unc_lines


def generate_random_boxes(h_field, w_field, beta_decay):

    img = np.zeros((h_field,w_field), np.uint8)
    num_boxes =  np.random.randint(30, size=1)

    for i in range(num_boxes[0]):

        x1, y1 = np.random.randint(w_field*0.9, size=1)[0], np.random.randint(h_field*0.9, size=1)[0]
        b = np.random.randint(np.minimum(h_field,w_field)*0.1, np.minimum(h_field,w_field)*0.3, size=1)[0]
        x2,y2 = x1+b,y1+int(b*(h_field/w_field))

        img = cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)

    img = np.where(img==0,beta_decay, 1)

    return img


def partialupdate(area, uncertainty):
    return area * uncertainty + 1 - area