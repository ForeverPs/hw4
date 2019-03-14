import cv2
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gaussian(size, u = 0, sigma = 1.5):
	filters = np.zeros((size[0], size[1]))
	k = int(size[0] / 2)
	for i in range(size[0]):
		for j in range(size[1]):
			filters[i, j] = g(u, sigma, i, j, k)
	evid = 1.0 / filters[0, 0]
	for i in range(size[0]):
		for j in range(size[1]):
			filters[i, j] = int(filters[i, j] * evid)
	return filters / np.sum(filters)

def g(u, sigma, i, j, k):
	weight = math.exp(((-(i-k)**2-(j-k)**2)/2/sigma**2))/2/math.pi/sigma**2
	return weight

def p_gaussian(img, filters):
	temp = img.copy()
	temp = signal.convolve2d(temp, filters, boundary='symm', mode='same')
	return temp

def p_sobel(padding, begin_x, begin_y, m, n):
	new = padding.copy()
	fy = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])
	fx = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
	for i in range(begin_x-1, begin_x+m):
		for j in range(begin_y-1, begin_y+n):
			dy = np.sum(np.multiply(fy, padding[i-1:i+2,j-1:j+2]))
			dx = np.sum(np.multiply(fx, padding[i-1:i+2,j-1:j+2]))
			new[i,j] = round(np.sqrt(dx**2 + dy**2))
	return new[begin_x-1:begin_x+m,begin_y-1:begin_y+n]

def process(filename, filter_name, size):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	if filter_name == 'gaussian':
		filters = gaussian(size)
		return p_gaussian(img, filters), img
	elif filter_name == 'middle':
		return signal.medfilt2d(img, size), img
	elif filter_name == 'unsharp':
		k = float(input('Please input weight k:\n'))
		blurfilters = gaussian(size)
		blur = p_gaussian(img, blurfilters)
		mask = img - blur
		return img + k * mask, img
	elif filter_name == 'sobel':
		m, n = np.shape(img)
		temp = np.pad(img, ((size[0]-1, size[0]-1), (size[1]-1, size[1]-1)), 'symmetric')
		new = p_sobel(temp, size[0]-1, size[1]-1, m, n)
		return new, img
	elif filter_name == 'laplace':
		filters = np.matrix([[0,1,0],[1,-4,1],[0,1,0]])
		lap = signal.convolve2d(img.copy(), filters, boundary='symm', mode='same').astype(int)
		lap[lap<0]=0
		return lap, img
	elif filter_name == 'canny':
		return cv2.Canny(img.copy(),20,200), img

def draw(names, images, row, col, c='blue'):
	assert isinstance(names, list) and isinstance(images, list), 'unexpected data type'
	assert len(names) == len(images), 'dimension mismatch'
	plt.figure('SHOW')
	for i in range(len(names)):
		plt.subplot(row, col, i+1)
		plt.title(names[i])
		plt.imshow(images[i], cmap='gray')
	plt.show()



if __name__ == '__main__':
#	filename = 'test4.tif'
#	size=[3,3]
#	new1, _1 = process(filename, 'gaussian', size)
#	new2, _2 = process(filename, 'middle', size)
#	draw(['GAUSSIAN','MEDIAN'],[new1,new2],row=1,col=2)
#	canny, orig = process(filename, 'canny', size)
#	draw(['CANNY','ORIGINAL'],[canny, orig],row=1,col=2)
